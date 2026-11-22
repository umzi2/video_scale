//! Splitting a frame into tiles so one model pass fits in GPU memory.
//!
//! The frame is divided into as few tiles as the allowed tile size permits.
//! Neighbouring tiles overlap, and inside the overlap the two tiles contribute
//! `sin²`/`cos²` weights that sum to one, so no seam is visible. Tile sizes
//! and interior offsets use eight-pixel alignment where coverage permits.
//!
//! The layout depends only on the frame geometry, so it is computed once and the
//! model session is compiled for that tile size.

/// Tile edges and offsets are multiples of this many pixels where possible.
const ALIGN: usize = 8;

/// One tile: where it comes from and how its upscaled pixels are blended.
pub struct Tile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    /// Blend weight per output pixel, row major, `2 * height` by `2 * width`.
    pub weights: Vec<f32>,
}

/// Tile layout of one frame.
pub struct Plan {
    columns: Axis,
    rows: Axis,
}

impl Plan {
    /// Plans tiles no larger than `max_tile` pixels per edge.
    pub fn new(width: usize, height: usize, max_tile: usize) -> Self {
        Self {
            columns: Axis::new(width, max_tile),
            rows: Axis::new(height, max_tile),
        }
    }

    /// Size of every tile; the frame is at least this large per axis.
    pub fn tile_size(&self) -> (usize, usize) {
        (self.columns.tile, self.rows.tile)
    }

    /// Whether the whole frame fits into one tile, making blending unnecessary.
    pub fn is_single_tile(&self) -> bool {
        self.columns.starts.len() == 1 && self.rows.starts.len() == 1
    }

    /// All tiles, row by row, with their blend weights.
    pub fn tiles(&self) -> Vec<Tile> {
        let per_column: Vec<Vec<f32>> = (0..self.columns.starts.len())
            .map(|index| self.columns.weights(index))
            .collect();
        let mut tiles = Vec::with_capacity(per_column.len() * self.rows.starts.len());
        for (row, &y) in self.rows.starts.iter().enumerate() {
            let per_row = self.rows.weights(row);
            for (column, &x) in self.columns.starts.iter().enumerate() {
                let mut weights = Vec::with_capacity(per_row.len() * per_column[column].len());
                // Row major over the upscaled tile: row weight times column weight.
                for row_weight in &per_row {
                    for column_weight in &per_column[column] {
                        weights.push(row_weight * column_weight);
                    }
                }
                tiles.push(Tile {
                    x,
                    y,
                    width: self.columns.tile,
                    height: self.rows.tile,
                    weights,
                });
            }
        }
        tiles
    }
}

/// One axis of the plan: tile size and the offsets its tiles start at.
struct Axis {
    starts: Vec<usize>,
    tile: usize,
}

impl Axis {
    fn new(dim: usize, max_tile: usize) -> Self {
        let limit = max_tile.max(ALIGN).min(dim);
        let tile = if limit >= dim {
            dim
        } else {
            limit / ALIGN * ALIGN
        };
        Self {
            starts: starts(dim, tile),
            tile,
        }
    }

    /// Blend weight of every output pixel of tile `index` along this axis.
    fn weights(&self, index: usize) -> Vec<f32> {
        let length = 2 * self.tile;
        let mut weights = vec![1.0; length];
        if let Some(previous) = index.checked_sub(1) {
            let overlap = 2 * (self.tile - (self.starts[index] - self.starts[previous]));
            for (position, weight) in weights.iter_mut().take(overlap.min(length)).enumerate() {
                *weight = fade(position, overlap);
            }
        }
        if let Some(next) = self.starts.get(index + 1) {
            let overlap = (2 * (self.starts[index] + self.tile - next)).min(length);
            for offset in 0..overlap {
                weights[length - overlap + offset] *= 1.0 - fade(offset, overlap);
            }
        }
        weights
    }
}

/// `sin²` ramp over an overlap of `length` pixels: 0 before it, 1 after it.
///
/// Two neighbouring tiles use this and its complement, so the pair sums to one
/// at every shared pixel.
fn fade(position: usize, length: usize) -> f32 {
    let step = (position as f32 + 0.5) / length as f32;
    (step * std::f32::consts::FRAC_PI_2).sin().powi(2)
}

/// Offsets where tiles start along one axis.
fn starts(dim: usize, tile: usize) -> Vec<usize> {
    if tile >= dim {
        return vec![0];
    }
    let span = dim - tile;
    // Each additional tile must share at least one pixel with its predecessor.
    let stride = tile - 1;
    let steps = span.div_ceil(stride);
    let mut starts = Vec::with_capacity(steps + 1);
    starts.push(0);
    for index in 1..steps {
        // Stay within one step of the previous tile and within reach of the end.
        let previous = starts[index - 1];
        let lower = (previous + 1).max(span - (steps - index) * stride);
        let upper = (previous + stride).min(span - (steps - index));
        let ideal = ((span as u128 * index as u128) / steps as u128) as usize;
        let first_aligned = lower.next_multiple_of(ALIGN);
        let last_aligned = upper / ALIGN * ALIGN;
        starts.push(if first_aligned <= last_aligned {
            (ideal / ALIGN * ALIGN).clamp(first_aligned, last_aligned)
        } else {
            ideal.clamp(lower, upper)
        });
    }
    starts.push(span);
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_the_frame_with_the_fewest_tiles() {
        for (width, height) in [(640, 360), (1920, 1080), (100, 100), (13, 9), (256, 256)] {
            for max_tile in [64, 128, 256, 512, 4096] {
                let plan = Plan::new(width, height, max_tile);
                let (tile_w, tile_h) = plan.tile_size();
                assert!(tile_w <= width && tile_h <= height);
                if tile_w < width {
                    assert_eq!(tile_w % ALIGN, 0, "{width} {max_tile}");
                }
                let columns = 1 + (width - tile_w).div_ceil(tile_w - 1);
                let rows = 1 + (height - tile_h).div_ceil(tile_h - 1);
                let tiles = plan.tiles();
                assert_eq!(
                    tiles.len(),
                    columns * rows,
                    "{width}x{height} max {max_tile}"
                );
                assert!(tiles.iter().all(|tile| tile.x + tile.width <= width));
                assert!(tiles.iter().all(|tile| tile.y + tile.height <= height));
            }
        }
    }

    #[test]
    fn axis_starts_cover_exact_multiples_and_nearby_lengths() {
        assert_eq!(starts(640, 256), [0, 192, 384]);
        assert_eq!(starts(360, 256), [0, 104]);
        assert_eq!(starts(511, 256), [0, 255]);
        assert_eq!(starts(512, 256), [0, 128, 256]);
        assert_eq!(starts(766, 256), [0, 255, 510]);
        for tile in [8, 16, 64, 256] {
            for dim in tile + 1..=4 * tile + 1 {
                let offsets = starts(dim, tile);
                assert_eq!(offsets[0], 0);
                assert_eq!(offsets.last(), Some(&(dim - tile)));
                assert!(
                    offsets
                        .windows(2)
                        .all(|p| p[0] < p[1] && p[1] - p[0] < tile)
                );
                assert!(tile + (offsets.len() - 2) * (tile - 1) < dim);
            }
        }
    }

    #[test]
    fn overlaps_fade_to_one_weight_per_pixel() {
        for (width, height, max_tile) in [(640, 360, 256), (1920, 1080, 512), (300, 200, 128)] {
            let plan = Plan::new(width, height, max_tile);
            let mut weights = vec![0.0f32; 4 * width * height];
            for tile in plan.tiles() {
                for row in 0..2 * tile.height {
                    for column in 0..2 * tile.width {
                        let weight = tile.weights[row * 2 * tile.width + column];
                        let (output_row, output_column) = (2 * tile.y + row, 2 * tile.x + column);
                        weights[output_row * 2 * width + output_column] += weight;
                    }
                }
            }
            for (index, weight) in weights.iter().enumerate() {
                assert!(
                    (weight - 1.0).abs() < 1e-5,
                    "{width}x{height} max {max_tile}: pixel {index} weighs {weight}"
                );
            }
        }
    }

    #[test]
    fn axis_weights_change_gradually() {
        for (dim, max_tile) in [(640, 256), (1920, 512), (360, 128)] {
            let axis = Axis::new(dim, max_tile);
            for index in 0..axis.starts.len() {
                let weights = axis.weights(index);
                for pair in weights.windows(2) {
                    assert!(
                        (pair[1] - pair[0]).abs() < 0.1,
                        "jump from {} to {} in {dim}/{max_tile} tile {index}",
                        pair[0],
                        pair[1]
                    );
                }
                assert!(weights.iter().all(|weight| (0.0..=1.0).contains(weight)));
            }
        }
    }

    #[test]
    fn single_tile_when_the_limit_covers_the_frame() {
        let plan = Plan::new(320, 180, 512);
        assert!(plan.is_single_tile());
        assert_eq!(plan.tile_size(), (320, 180));
        let tiles = plan.tiles();
        assert_eq!(tiles.len(), 1);
        assert!(tiles[0].weights.iter().all(|weight| *weight == 1.0));
    }

    #[test]
    fn tiny_frames_stay_one_tile() {
        let plan = Plan::new(5, 3, 64);
        assert!(plan.is_single_tile());
        assert_eq!(plan.tile_size(), (5, 3));
    }
}
