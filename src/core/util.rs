use crate::{Error, Result};

pub(crate) fn next_fast_len(target: usize) -> usize {
    if target <= 1 {
        return 1;
    }

    let mut candidate = target;
    while !is_235_smooth(candidate) {
        candidate += 1;
    }
    candidate
}

pub(crate) fn checked_volume_2d(height: usize, width: usize) -> Result<usize> {
    height.checked_mul(width).ok_or(Error::InvalidParameter)
}

pub(crate) fn checked_volume_3d(depth: usize, height: usize, width: usize) -> Result<usize> {
    checked_volume_2d(depth, height)?
        .checked_mul(width)
        .ok_or(Error::InvalidParameter)
}

fn is_235_smooth(mut value: usize) -> bool {
    for prime in [2, 3, 5] {
        while value.is_multiple_of(prime) {
            value /= prime;
        }
    }
    value == 1
}

#[cfg(test)]
mod tests {
    use super::next_fast_len;

    #[test]
    fn next_fast_len_returns_235_smooth_size() {
        assert_eq!(next_fast_len(1), 1);
        assert_eq!(next_fast_len(6), 6);
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(next_fast_len(17), 18);
        assert_eq!(next_fast_len(10007), 10125);
    }
}
