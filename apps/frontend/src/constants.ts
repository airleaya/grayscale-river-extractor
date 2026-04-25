import type { PipelineConfig } from './types'

/**
 * Provide one stable default request so the UI can stay small and the create
 * flow remains deterministic during backend and algorithm scaffolding work.
 */
export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  preprocess: {
    height_mapping: 'bright_is_high',
    smooth: true,
    smooth_kernel_size: 3,
    fill_sinks: true,
    fill_sink_algorithm: 'auto',
    max_fill_depth: null,
    deep_basin_mode: 'mark',
    fast_fill_min_pixels: 8_000_000,
    preserve_nodata: true,
    nodata_value: null,
    use_auto_mask: false,
    auto_mask_border_sensitivity: 1.0,
    auto_mask_texture_sensitivity: 1.0,
    auto_mask_min_region_size: 2048,
    use_mask: false,
  },
  flow_direction: {
    method: 'D8',
    use_rust_kernel: false,
    slope_weight: 1.0,
    flat_escape_weight: 0.6,
    outlet_proximity_weight: 0.4,
    continuity_weight: 0.3,
    flat_outlet_length_weight: 0.35,
    flat_outlet_distance_weight: 1.5,
  },
  flow_accumulation: {
    normalize: true,
    use_rust_kernel: true,
  },
  channel_extract: {
    accumulation_threshold: 200,
    channel_length_threshold: 1,
  },
  save_intermediates: true,
  total_tiles: 72,
  preview_max_side: 4096,
}
