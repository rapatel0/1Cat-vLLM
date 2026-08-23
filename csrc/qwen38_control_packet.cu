// SPDX-License-Identifier: Apache-2.0
// Fixed-shape Qwen3.8/V100 control-packet helpers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

namespace {

__global__ void scatter_block_tables_kernel(
    const int32_t* __restrict__ packet, int32_t* __restrict__ dst0,
    int32_t* __restrict__ dst1, int32_t* __restrict__ dst2,
    int32_t* __restrict__ dst3, int width0, int width1, int width2,
    int width3) {
  const int index = threadIdx.x;
  if (index < width0) {
    dst0[index] = packet[index];
    return;
  }
  int offset = width0;
  if (index < offset + width1) {
    dst1[index - offset] = packet[index];
    return;
  }
  offset += width1;
  if (index < offset + width2) {
    dst2[index - offset] = packet[index];
    return;
  }
  offset += width2;
  if (index < offset + width3) {
    dst3[index - offset] = packet[index];
  }
}

// Exact batch-one Qwen3.8 verifier packet. All host values travel as int64 so
// mixed int32/int64/bool destinations still require only one pinned H2D.
template <typename DraftT>
__global__ void scatter_prepare_inputs_kernel(
    const int64_t* __restrict__ packet, int32_t* __restrict__ bt0,
    int32_t* __restrict__ bt1, int32_t* __restrict__ bt2,
    int32_t* __restrict__ bt3, int32_t* __restrict__ query_start,
    bool* __restrict__ discard, int32_t* __restrict__ accepted,
    int32_t* __restrict__ selector, int64_t* __restrict__ prev_positions,
    int32_t* __restrict__ prev_drafts,
    int32_t* __restrict__ cpu_num_computed_values,
    int64_t* __restrict__ req_indices, int64_t* __restrict__ query_pos,
    int32_t* __restrict__ num_scheduled, int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ prev_sampled_token_ids,
    int64_t prev_sampled_stride,
    const DraftT* __restrict__ draft_token_ids,
    int32_t* __restrict__ grouped_state_ids,
    int64_t grouped_state_group_stride,
    int32_t* __restrict__ output_state_ids,
    int64_t output_state_group_stride,
    int64_t output_state_row_stride,
    bool* __restrict__ output_sequence_masks,
    int64_t output_sequence_mask_group_stride,
    int32_t* __restrict__ output_token_indices,
    int64_t output_token_index_group_stride,
    int32_t* __restrict__ output_query_start_loc,
    int64_t output_query_start_group_stride,
    int32_t* __restrict__ output_accepted_tokens,
    int64_t output_accepted_group_stride,
    int32_t* __restrict__ output_state_selectors,
    int64_t output_selector_group_stride,
    bool build_gdn_metadata) {
  const int index = threadIdx.x;
  if (index < 44) {
    bt0[index] = static_cast<int32_t>(packet[index]);
  } else if (index < 88) {
    bt1[index - 44] = static_cast<int32_t>(packet[index]);
  } else if (index < 132) {
    bt2[index - 88] = static_cast<int32_t>(packet[index]);
  } else if (index < 173) {
    bt3[index - 132] = static_cast<int32_t>(packet[index]);
  } else if (index < 178) {
    query_start[index - 173] = static_cast<int32_t>(packet[index]);
  } else if (index < 182) {
    discard[index - 178] = packet[index] != 0;
  } else if (index < 186) {
    accepted[index - 182] = static_cast<int32_t>(packet[index]);
  } else if (index < 190) {
    selector[index - 186] = static_cast<int32_t>(packet[index]);
  } else if (index < 194) {
    prev_positions[index - 190] = packet[index];
  } else if (index < 198) {
    prev_drafts[index - 194] = static_cast<int32_t>(packet[index]);
  } else if (index < 202) {
    cpu_num_computed_values[index - 198] = static_cast<int32_t>(packet[index]);
  } else if (index < 206) {
    req_indices[index - 202] = packet[index];
  } else if (index < 210) {
    query_pos[index - 206] = packet[index];
  } else if (index < 214) {
    num_scheduled[index - 210] = static_cast<int32_t>(packet[index]);
  } else if (index == 214) {
    // The exact q=4 executor always has one request carried from the previous
    // async-scheduling batch.  Consume its mandatory sampled token directly
    // from the persistent GPU result instead of uploading temporary gather
    // indices from Python.
    const int64_t prev_index = packet[190];
    input_ids[0] = prev_sampled_token_ids[prev_index * prev_sampled_stride];
  } else if (index < 218) {
    // Native MTP stores three drafts per prior request in a contiguous tensor.
    // The packet's previous-position mapping selects the matching row.
    const int64_t prev_index = packet[190];
    input_ids[index - 214] = static_cast<int32_t>(
        draft_token_ids[prev_index * 3 + (index - 215)]);
  }

  // The exact Qwen3.8 q=4 verifier has three GDN cache groups, four
  // recurrent-state slots, and one live request padded to four token rows.
  // Materialize all graph-persistent GDN metadata from the same control
  // packet that already feeds the verifier inputs.  This replaces a second
  // pinned H2D and the grouped Triton metadata launch without changing the
  // generic path or any acceptance/state-selection semantics.
  if (build_gdn_metadata && index < 12) {
    constexpr int kNumGroups = 3;
    constexpr int kStateWidth = 4;
    constexpr int kBatchSize = 4;
    constexpr int kStatePacketOffset = 214;
    constexpr int kPadSlotId = -1;
    const int group = index / kBatchSize;
    const int row = index % kBatchSize;
    const bool live_row = row == 0;
    if (group < kNumGroups) {
      for (int col = 0; col < kStateWidth; ++col) {
        const int32_t state_id = live_row
                                     ? static_cast<int32_t>(packet[
                                           kStatePacketOffset +
                                           group * kStateWidth + col])
                                     : kPadSlotId;
        output_state_ids[group * output_state_group_stride +
                         row * output_state_row_stride + col] = state_id;
        if (live_row) {
          grouped_state_ids[group * grouped_state_group_stride + col] =
              state_id;
        }
      }
      output_sequence_masks[group * output_sequence_mask_group_stride + row] =
          live_row;
      output_accepted_tokens[group * output_accepted_group_stride + row] =
          live_row ? static_cast<int32_t>(packet[182]) : 1;
      output_state_selectors[group * output_selector_group_stride + row] =
          live_row ? static_cast<int32_t>(packet[186]) : 1;
      output_token_indices[group * output_token_index_group_stride + row] = row;
      output_query_start_loc[group * output_query_start_group_stride + row] =
          live_row ? 0 : kBatchSize;
      if (live_row) {
        output_query_start_loc[group * output_query_start_group_stride +
                               kBatchSize] = kBatchSize;
      }
    }
  }
}

__global__ void prebuild_flash_q4_metadata_kernel(
    const int32_t* __restrict__ source_block_table,
    const int32_t* __restrict__ source_seq_lens,
    int32_t* __restrict__ output_block_table,
    int64_t output_block_table_stride,
    int32_t* __restrict__ output_seq_lens,
    int32_t* __restrict__ output_query_start_loc,
    int num_block_cols) {
  const int block_col = threadIdx.x;
  if (block_col < num_block_cols) {
    const int32_t block = max(source_block_table[block_col], 0);
#pragma unroll
    for (int query_idx = 0; query_idx < 4; ++query_idx) {
      output_block_table[query_idx * output_block_table_stride + block_col] =
          block;
    }
  }
  if (block_col < 4) {
    const int32_t seq_len = max(source_seq_lens[0], 4);
    output_seq_lens[block_col] = seq_len - 4 + block_col + 1;
  }
  if (block_col == 0) {
    output_query_start_loc[0] = 0;
    output_query_start_loc[1] = 4;
  }
}

void scatter_block_tables(at::Tensor packet, at::Tensor dst0, at::Tensor dst1,
                          at::Tensor dst2, at::Tensor dst3, int64_t width0,
                          int64_t width1, int64_t width2, int64_t width3) {
  TORCH_CHECK(packet.is_cuda(), "control packet must be CUDA-resident");
  TORCH_CHECK(packet.scalar_type() == at::ScalarType::Int,
              "control packet must be int32");
  TORCH_CHECK(packet.is_contiguous(), "control packet must be contiguous");
  TORCH_CHECK(width0 > 0 && width1 > 0 && width2 > 0 && width3 > 0,
              "block-table widths must be positive");
  const int64_t total = width0 + width1 + width2 + width3;
  TORCH_CHECK(total <= 256, "control packet exceeds one SM70 block");
  TORCH_CHECK(packet.numel() >= total, "control packet is too small");

  const at::Tensor destinations[] = {dst0, dst1, dst2, dst3};
  const int64_t widths[] = {width0, width1, width2, width3};
  for (int index = 0; index < 4; ++index) {
    const auto& dst = destinations[index];
    TORCH_CHECK(dst.is_cuda() && dst.device() == packet.device(),
                "block-table destination must be on the packet device");
    TORCH_CHECK(dst.scalar_type() == at::ScalarType::Int,
                "block-table destination must be int32");
    TORCH_CHECK(dst.is_contiguous(),
                "block-table destination must be contiguous");
    TORCH_CHECK(dst.numel() >= widths[index],
                "block-table destination is too small");
  }

  const c10::cuda::OptionalCUDAGuard device_guard(packet.device());
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  scatter_block_tables_kernel<<<1, 256, 0, stream>>>(
      packet.data_ptr<int32_t>(), dst0.data_ptr<int32_t>(),
      dst1.data_ptr<int32_t>(), dst2.data_ptr<int32_t>(),
      dst3.data_ptr<int32_t>(), static_cast<int>(width0),
      static_cast<int>(width1), static_cast<int>(width2),
      static_cast<int>(width3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_prepare_inputs(
    at::Tensor packet, at::Tensor bt0, at::Tensor bt1, at::Tensor bt2,
    at::Tensor bt3, at::Tensor query_start, at::Tensor discard,
    at::Tensor accepted, at::Tensor selector, at::Tensor prev_positions,
    at::Tensor prev_drafts, at::Tensor cpu_num_computed_values,
    at::Tensor req_indices, at::Tensor query_pos, at::Tensor num_scheduled,
    at::Tensor input_ids, at::Tensor prev_sampled_token_ids,
    int64_t prev_sampled_stride, at::Tensor draft_token_ids) {
  TORCH_CHECK(packet.is_cuda() && packet.scalar_type() == at::ScalarType::Long &&
                  packet.is_contiguous() && packet.numel() == 214,
              "Qwen3.8 prepare packet must be contiguous CUDA int64[214]");
  const at::Tensor destinations[] = {
      bt0,          bt1,         bt2,          bt3,       query_start,
      discard,      accepted,    selector,     prev_positions,
      prev_drafts,  cpu_num_computed_values, req_indices, query_pos,
      num_scheduled, input_ids, prev_sampled_token_ids, draft_token_ids};
  for (const auto& destination : destinations) {
    TORCH_CHECK(destination.is_cuda() &&
                    destination.device() == packet.device() &&
                    destination.is_contiguous(),
                "Qwen3.8 prepare destination must be contiguous on packet device");
  }
  TORCH_CHECK(bt0.numel() >= 44 && bt1.numel() >= 44 && bt2.numel() >= 44 &&
                  bt3.numel() >= 41 && query_start.numel() >= 5 &&
                  discard.numel() >= 4 && accepted.numel() >= 4 &&
                  selector.numel() >= 4 && prev_positions.numel() >= 4 &&
                  prev_drafts.numel() >= 4 &&
                  cpu_num_computed_values.numel() >= 4 &&
                  req_indices.numel() >= 4 && query_pos.numel() >= 4 &&
                  num_scheduled.numel() >= 4 && input_ids.numel() >= 4 &&
                  prev_sampled_token_ids.numel() >= 1 &&
                  draft_token_ids.numel() >= 3,
              "Qwen3.8 prepare destination is too small");
  TORCH_CHECK(bt0.scalar_type() == at::ScalarType::Int &&
                  bt1.scalar_type() == at::ScalarType::Int &&
                  bt2.scalar_type() == at::ScalarType::Int &&
                  bt3.scalar_type() == at::ScalarType::Int &&
                  query_start.scalar_type() == at::ScalarType::Int &&
                  discard.scalar_type() == at::ScalarType::Bool &&
                  accepted.scalar_type() == at::ScalarType::Int &&
                  selector.scalar_type() == at::ScalarType::Int &&
                  prev_positions.scalar_type() == at::ScalarType::Long &&
                  prev_drafts.scalar_type() == at::ScalarType::Int &&
                  cpu_num_computed_values.scalar_type() == at::ScalarType::Int &&
                  req_indices.scalar_type() == at::ScalarType::Long &&
                  query_pos.scalar_type() == at::ScalarType::Long &&
                  num_scheduled.scalar_type() == at::ScalarType::Int &&
                  input_ids.scalar_type() == at::ScalarType::Int &&
                  prev_sampled_token_ids.scalar_type() == at::ScalarType::Int &&
                  (draft_token_ids.scalar_type() == at::ScalarType::Int ||
                   draft_token_ids.scalar_type() == at::ScalarType::Long),
              "Qwen3.8 prepare destination dtype mismatch");
  TORCH_CHECK(prev_sampled_stride > 0,
              "Qwen3.8 previous sampled-token stride must be positive");
  const c10::cuda::OptionalCUDAGuard device_guard(packet.device());
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (draft_token_ids.scalar_type() == at::ScalarType::Long) {
    scatter_prepare_inputs_kernel<int64_t><<<1, 256, 0, stream>>>(
        packet.data_ptr<int64_t>(), bt0.data_ptr<int32_t>(),
        bt1.data_ptr<int32_t>(), bt2.data_ptr<int32_t>(),
        bt3.data_ptr<int32_t>(), query_start.data_ptr<int32_t>(),
        discard.data_ptr<bool>(), accepted.data_ptr<int32_t>(),
        selector.data_ptr<int32_t>(), prev_positions.data_ptr<int64_t>(),
        prev_drafts.data_ptr<int32_t>(),
        cpu_num_computed_values.data_ptr<int32_t>(),
        req_indices.data_ptr<int64_t>(), query_pos.data_ptr<int64_t>(),
        num_scheduled.data_ptr<int32_t>(), input_ids.data_ptr<int32_t>(),
        prev_sampled_token_ids.data_ptr<int32_t>(), prev_sampled_stride,
        draft_token_ids.data_ptr<int64_t>(), nullptr, 0, nullptr, 0, 0,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, false);
  } else {
    scatter_prepare_inputs_kernel<int32_t><<<1, 256, 0, stream>>>(
        packet.data_ptr<int64_t>(), bt0.data_ptr<int32_t>(),
        bt1.data_ptr<int32_t>(), bt2.data_ptr<int32_t>(),
        bt3.data_ptr<int32_t>(), query_start.data_ptr<int32_t>(),
        discard.data_ptr<bool>(), accepted.data_ptr<int32_t>(),
        selector.data_ptr<int32_t>(), prev_positions.data_ptr<int64_t>(),
        prev_drafts.data_ptr<int32_t>(),
        cpu_num_computed_values.data_ptr<int32_t>(),
        req_indices.data_ptr<int64_t>(), query_pos.data_ptr<int64_t>(),
        num_scheduled.data_ptr<int32_t>(), input_ids.data_ptr<int32_t>(),
        prev_sampled_token_ids.data_ptr<int32_t>(), prev_sampled_stride,
        draft_token_ids.data_ptr<int32_t>(), nullptr, 0, nullptr, 0, 0,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_prepare_inputs_and_gdn_metadata(
    at::Tensor packet, at::Tensor bt0, at::Tensor bt1, at::Tensor bt2,
    at::Tensor bt3, at::Tensor query_start, at::Tensor discard,
    at::Tensor accepted, at::Tensor selector, at::Tensor prev_positions,
    at::Tensor prev_drafts, at::Tensor cpu_num_computed_values,
    at::Tensor req_indices, at::Tensor query_pos, at::Tensor num_scheduled,
    at::Tensor input_ids, at::Tensor prev_sampled_token_ids,
    int64_t prev_sampled_stride, at::Tensor draft_token_ids,
    at::Tensor grouped_state_ids, at::Tensor output_state_ids,
    at::Tensor output_sequence_masks, at::Tensor output_token_indices,
    at::Tensor output_query_start_loc, at::Tensor output_accepted_tokens,
    at::Tensor output_state_selectors) {
  TORCH_CHECK(packet.is_cuda() && packet.scalar_type() == at::ScalarType::Long &&
                  packet.is_contiguous() && packet.numel() == 226,
              "Qwen3.8 GDN prepare packet must be contiguous CUDA int64[226]");
  const at::Tensor destinations[] = {
      bt0, bt1, bt2, bt3, query_start, discard, accepted, selector,
      prev_positions, prev_drafts, cpu_num_computed_values, req_indices,
      query_pos, num_scheduled, input_ids, prev_sampled_token_ids,
      draft_token_ids, grouped_state_ids, output_state_ids,
      output_sequence_masks, output_token_indices, output_query_start_loc,
      output_accepted_tokens, output_state_selectors};
  for (const auto& destination : destinations) {
    TORCH_CHECK(destination.is_cuda() &&
                    destination.device() == packet.device() &&
                    destination.is_contiguous(),
                "Qwen3.8 GDN prepare tensor must be contiguous on packet device");
  }
  TORCH_CHECK(bt0.scalar_type() == at::ScalarType::Int && bt0.numel() >= 44 &&
                  bt1.scalar_type() == at::ScalarType::Int && bt1.numel() >= 44 &&
                  bt2.scalar_type() == at::ScalarType::Int && bt2.numel() >= 44 &&
                  bt3.scalar_type() == at::ScalarType::Int && bt3.numel() >= 41 &&
                  query_start.scalar_type() == at::ScalarType::Int &&
                  query_start.numel() >= 5 &&
                  discard.scalar_type() == at::ScalarType::Bool &&
                  discard.numel() >= 4 &&
                  accepted.scalar_type() == at::ScalarType::Int &&
                  accepted.numel() >= 4 &&
                  selector.scalar_type() == at::ScalarType::Int &&
                  selector.numel() >= 4 &&
                  prev_positions.scalar_type() == at::ScalarType::Long &&
                  prev_positions.numel() >= 4 &&
                  prev_drafts.scalar_type() == at::ScalarType::Int &&
                  prev_drafts.numel() >= 4 &&
                  cpu_num_computed_values.scalar_type() == at::ScalarType::Int &&
                  cpu_num_computed_values.numel() >= 4 &&
                  req_indices.scalar_type() == at::ScalarType::Long &&
                  req_indices.numel() >= 4 &&
                  query_pos.scalar_type() == at::ScalarType::Long &&
                  query_pos.numel() >= 4 &&
                  num_scheduled.scalar_type() == at::ScalarType::Int &&
                  num_scheduled.numel() >= 4 &&
                  input_ids.scalar_type() == at::ScalarType::Int &&
                  input_ids.numel() >= 4 &&
                  prev_sampled_token_ids.scalar_type() == at::ScalarType::Int &&
                  prev_sampled_token_ids.numel() >= 1 &&
                  (draft_token_ids.scalar_type() == at::ScalarType::Int ||
                   draft_token_ids.scalar_type() == at::ScalarType::Long) &&
                  draft_token_ids.numel() >= 3,
              "Qwen3.8 GDN prepare input contract mismatch");
  TORCH_CHECK(grouped_state_ids.scalar_type() == at::ScalarType::Int &&
                  grouped_state_ids.dim() == 3 &&
                  grouped_state_ids.size(0) >= 3 &&
                  grouped_state_ids.size(1) >= 1 &&
                  grouped_state_ids.size(2) >= 4 &&
                  output_state_ids.scalar_type() == at::ScalarType::Int &&
                  output_state_ids.dim() == 3 && output_state_ids.size(0) == 3 &&
                  output_state_ids.size(1) >= 4 && output_state_ids.size(2) >= 4 &&
                  output_sequence_masks.scalar_type() == at::ScalarType::Bool &&
                  output_sequence_masks.dim() == 2 &&
                  output_sequence_masks.size(0) == 3 &&
                  output_sequence_masks.size(1) >= 4 &&
                  output_token_indices.scalar_type() == at::ScalarType::Int &&
                  output_token_indices.dim() == 2 &&
                  output_token_indices.size(0) == 3 &&
                  output_token_indices.size(1) >= 4 &&
                  output_query_start_loc.scalar_type() == at::ScalarType::Int &&
                  output_query_start_loc.dim() == 2 &&
                  output_query_start_loc.size(0) == 3 &&
                  output_query_start_loc.size(1) >= 5 &&
                  output_accepted_tokens.scalar_type() == at::ScalarType::Int &&
                  output_accepted_tokens.dim() == 2 &&
                  output_accepted_tokens.size(0) == 3 &&
                  output_accepted_tokens.size(1) >= 4 &&
                  output_state_selectors.scalar_type() == at::ScalarType::Int &&
                  output_state_selectors.dim() == 2 &&
                  output_state_selectors.size(0) == 3 &&
                  output_state_selectors.size(1) >= 4,
              "Qwen3.8 GDN prepare metadata contract mismatch");
  TORCH_CHECK(prev_sampled_stride > 0,
              "Qwen3.8 previous sampled-token stride must be positive");

  const c10::cuda::OptionalCUDAGuard device_guard(packet.device());
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_GDN_PREPARE(DRAFT_TYPE)                                         \
  scatter_prepare_inputs_kernel<DRAFT_TYPE><<<1, 256, 0, stream>>>(           \
      packet.data_ptr<int64_t>(), bt0.data_ptr<int32_t>(),                    \
      bt1.data_ptr<int32_t>(), bt2.data_ptr<int32_t>(),                       \
      bt3.data_ptr<int32_t>(), query_start.data_ptr<int32_t>(),               \
      discard.data_ptr<bool>(), accepted.data_ptr<int32_t>(),                 \
      selector.data_ptr<int32_t>(), prev_positions.data_ptr<int64_t>(),       \
      prev_drafts.data_ptr<int32_t>(),                                        \
      cpu_num_computed_values.data_ptr<int32_t>(),                            \
      req_indices.data_ptr<int64_t>(), query_pos.data_ptr<int64_t>(),         \
      num_scheduled.data_ptr<int32_t>(), input_ids.data_ptr<int32_t>(),       \
      prev_sampled_token_ids.data_ptr<int32_t>(), prev_sampled_stride,        \
      draft_token_ids.data_ptr<DRAFT_TYPE>(),                                 \
      grouped_state_ids.data_ptr<int32_t>(), grouped_state_ids.stride(0),     \
      output_state_ids.data_ptr<int32_t>(), output_state_ids.stride(0),       \
      output_state_ids.stride(1), output_sequence_masks.data_ptr<bool>(),     \
      output_sequence_masks.stride(0), output_token_indices.data_ptr<int32_t>(), \
      output_token_indices.stride(0), output_query_start_loc.data_ptr<int32_t>(), \
      output_query_start_loc.stride(0),                                       \
      output_accepted_tokens.data_ptr<int32_t>(),                             \
      output_accepted_tokens.stride(0),                                       \
      output_state_selectors.data_ptr<int32_t>(),                             \
      output_state_selectors.stride(0), true)
  if (draft_token_ids.scalar_type() == at::ScalarType::Long) {
    LAUNCH_GDN_PREPARE(int64_t);
  } else {
    LAUNCH_GDN_PREPARE(int32_t);
  }
#undef LAUNCH_GDN_PREPARE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prebuild_flash_q4_metadata(
    at::Tensor source_block_table, at::Tensor source_seq_lens,
    at::Tensor output_block_table, at::Tensor output_seq_lens,
    at::Tensor output_query_start_loc) {
  const at::Tensor tensors[] = {
      source_block_table,
      source_seq_lens,
      output_block_table,
      output_seq_lens,
      output_query_start_loc,
  };
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda() &&
                    tensor.device() == source_block_table.device() &&
                    tensor.scalar_type() == at::ScalarType::Int &&
                    tensor.is_contiguous(),
                "Qwen3.8 Flash q4 metadata tensors must be contiguous CUDA "
                "int32 tensors on one device");
  }
  TORCH_CHECK(source_block_table.dim() == 2 &&
                  source_block_table.size(0) == 1 &&
                  source_block_table.size(1) > 0 &&
                  source_block_table.size(1) <= 256 &&
                  source_seq_lens.dim() == 1 && source_seq_lens.numel() == 1 &&
                  output_block_table.dim() == 2 &&
                  output_block_table.size(0) >= 4 &&
                  output_block_table.size(1) == source_block_table.size(1) &&
                  output_seq_lens.dim() == 1 && output_seq_lens.numel() >= 4 &&
                  output_query_start_loc.dim() == 1 &&
                  output_query_start_loc.numel() >= 2,
              "Qwen3.8 Flash q4 metadata shape mismatch");
  const c10::cuda::OptionalCUDAGuard device_guard(source_block_table.device());
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  prebuild_flash_q4_metadata_kernel<<<1, 256, 0, stream>>>(
      source_block_table.data_ptr<int32_t>(),
      source_seq_lens.data_ptr<int32_t>(),
      output_block_table.data_ptr<int32_t>(), output_block_table.stride(0),
      output_seq_lens.data_ptr<int32_t>(),
      output_query_start_loc.data_ptr<int32_t>(),
      static_cast<int>(source_block_table.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prebuild_flash_q4_metadata_meta(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor) {}

}  // namespace

TORCH_LIBRARY(qwen38_control, library) {
  library.def(
      "scatter_block_tables(Tensor packet, Tensor(a!) dst0, Tensor(b!) dst1, "
      "Tensor(c!) dst2, Tensor(d!) dst3, int width0, int width1, int width2, "
      "int width3) -> ()");
  library.def(
      "scatter_prepare_inputs(Tensor packet, Tensor(a!) bt0, Tensor(b!) bt1, "
      "Tensor(c!) bt2, Tensor(d!) bt3, Tensor(e!) query_start, Tensor(f!) "
      "discard, Tensor(g!) accepted, Tensor(h!) selector, Tensor(i!) "
      "prev_positions, Tensor(j!) prev_drafts, Tensor(k!) cpu_num_computed_values, "
      "Tensor(l!) req_indices, Tensor(m!) query_pos, Tensor(n!) "
      "num_scheduled, Tensor(o!) input_ids, Tensor prev_sampled_token_ids, "
      "int prev_sampled_stride, Tensor draft_token_ids) -> ()");
  library.def(
      "scatter_prepare_inputs_and_gdn_metadata(Tensor packet, Tensor(a!) bt0, "
      "Tensor(b!) bt1, Tensor(c!) bt2, Tensor(d!) bt3, Tensor(e!) query_start, "
      "Tensor(f!) discard, Tensor(g!) accepted, Tensor(h!) selector, Tensor(i!) "
      "prev_positions, Tensor(j!) prev_drafts, Tensor(k!) cpu_num_computed_values, "
      "Tensor(l!) req_indices, Tensor(m!) query_pos, Tensor(n!) num_scheduled, "
      "Tensor(o!) input_ids, Tensor prev_sampled_token_ids, int prev_sampled_stride, "
      "Tensor draft_token_ids, Tensor(p!) grouped_state_ids, Tensor(q!) "
      "output_state_ids, Tensor(r!) output_sequence_masks, Tensor(s!) "
      "output_token_indices, Tensor(t!) output_query_start_loc, Tensor(u!) "
      "output_accepted_tokens, Tensor(v!) output_state_selectors) -> ()");
  library.def(
      "prebuild_flash_q4_metadata(Tensor source_block_table, Tensor "
      "source_seq_lens, Tensor(a!) output_block_table, Tensor(b!) "
      "output_seq_lens, Tensor(c!) output_query_start_loc) -> ()");
}

TORCH_LIBRARY_IMPL(qwen38_control, CUDA, library) {
  library.impl("scatter_block_tables", &scatter_block_tables);
  library.impl("scatter_prepare_inputs", &scatter_prepare_inputs);
  library.impl("scatter_prepare_inputs_and_gdn_metadata",
               &scatter_prepare_inputs_and_gdn_metadata);
  library.impl("prebuild_flash_q4_metadata", &prebuild_flash_q4_metadata);
}

TORCH_LIBRARY_IMPL(qwen38_control, Meta, library) {
  library.impl("prebuild_flash_q4_metadata",
               &prebuild_flash_q4_metadata_meta);
}
