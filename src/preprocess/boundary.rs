use crate::{Boundary, Error, Result};

pub(crate) fn map_index(index: i64, len: usize, boundary: Boundary) -> Result<Option<usize>> {
    if len == 0 {
        return Err(Error::InvalidParameter);
    }

    let len_i64 = to_i64(len)?;

    match boundary {
        Boundary::Zero => {
            if index < 0 || index >= len_i64 {
                Ok(None)
            } else {
                Ok(Some(to_usize(index)?))
            }
        }
        Boundary::Replicate => {
            let mapped = if index < 0 {
                0
            } else if index >= len_i64 {
                len_i64 - 1
            } else {
                index
            };
            Ok(Some(to_usize(mapped)?))
        }
        Boundary::Periodic => {
            let mapped = index.rem_euclid(len_i64);
            Ok(Some(to_usize(mapped)?))
        }
        Boundary::Reflect => {
            if len == 1 {
                return Ok(Some(0));
            }

            let period = 2_i64
                .checked_mul(len_i64 - 1)
                .ok_or(Error::InvalidParameter)?;
            let folded = index.rem_euclid(period);
            let mapped = if folded < len_i64 {
                folded
            } else {
                period - folded
            };
            Ok(Some(to_usize(mapped)?))
        }
        Boundary::Symmetric => {
            let period = 2_i64.checked_mul(len_i64).ok_or(Error::InvalidParameter)?;
            let folded = index.rem_euclid(period);
            let mapped = if folded < len_i64 {
                folded
            } else {
                period - 1 - folded
            };
            Ok(Some(to_usize(mapped)?))
        }
    }
}

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::DimensionMismatch)
}

fn to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::DimensionMismatch)
}
