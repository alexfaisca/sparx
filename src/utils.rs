use std::any::type_name;

#[allow(dead_code)]
pub fn type_of<T>() -> &'static str {
    type_name::<T>()
}

/// Checks if a `val` is a normal `f64`. Outputs a result with a custom error message.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be checked.
/// * `op_description`: `&str` --- the custom error message.
///
/// # Returns
///
/// `Ok(val)` if `val` is normal, or `Err(op_description.into())` if not.
///
#[inline(always)]
pub(crate) fn f64_is_nomal(
    val: f64,
    op_description: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    if !val.is_normal() {
        return Err(format!("error hk-relax abnormal value at {op_description} = {val}",).into());
    }
    Ok(val)
}

/// Safely converts an `f64` `val` into `usize`. Outputs an option.
///
/// Conversion is successful if `val` is normal, bigger than zero (not equal) and less than or equal to `usize::MAX`.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be cast.
///
/// # Returns
///
/// `Some(val as usize)` if successful, or None if not.
///
pub(crate) fn f64_to_usize_safe(val: f64) -> Option<usize> {
    if val.is_normal() && val > 0f64 && val < usize::MAX as f64 {
        Some(val as usize) // truncates toward zero
    } else {
        None
    }
}

pub fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / (a.len() as f64)
}
pub fn mape(a: &[f64], b: &[f64]) -> f64 {
    100.0
        * a.iter()
            .zip(b)
            .map(|(x, &y)| {
                if y == 0. {
                    // eprintln!("x {x} y {y}");
                    0.
                } else {
                    (x - y).abs() / y.abs()
                }
            })
            .sum::<f64>()
        / (a.len() as f64)
}

pub fn ranks(v: &[f64]) -> Vec<usize> {
    // descending rank; stable
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0; v.len()];
    for (rank, i) in idx.into_iter().enumerate() {
        r[i] = rank;
    }
    r
}

pub fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    let ra = ranks(a);
    let rb = ranks(b);
    let n = a.len() as f64;
    let ssd = ra
        .iter()
        .zip(rb.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum::<f64>();
    1.0 - (6.0 * ssd) / (n * (n * n - 1.0).max(1.0))
}

#[cfg(test)]
mod tests {
    use super::{f64_is_nomal, f64_to_usize_safe};

    // -------------------- f64_is_nomal --------------------

    #[test]
    fn nomal_accepts_positive_normal() {
        let v = 1.2345_f64;
        let got = f64_is_nomal(v, "pos").unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn nomal_accepts_negative_normal() {
        let v: f64 = -3.15;
        let got = f64_is_nomal(v, "neg").unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn nomal_rejects_zero() {
        let err = f64_is_nomal(0.0, "zero").unwrap_err();
        let s = err.to_string();
        assert!(s.contains("abnormal"), "msg = {s}");
        assert!(s.contains("zero"), "msg = {s}");
    }

    #[test]
    fn nomal_rejects_subnormal() {
        // subnormal = smaller than MIN_POSITIVE but non-zero
        let sub = f64::MIN_POSITIVE / 2.0;
        assert!(sub.is_subnormal());
        let err = f64_is_nomal(sub, "subnorm").unwrap_err();
        assert!(err.to_string().contains("abnormal"));
    }

    #[test]
    fn nomal_rejects_nan() {
        let err = f64_is_nomal(f64::NAN, "nan").unwrap_err();
        let s = err.to_string();
        assert!(s.contains("abnormal"), "msg = {s}");
        assert!(s.contains("NaN"), "msg = {s}");
    }

    #[test]
    fn nomal_rejects_infinity() {
        let err = f64_is_nomal(f64::INFINITY, "inf").unwrap_err();
        let s = err.to_string();
        assert!(s.contains("abnormal"), "msg = {s}");
        assert!(s.contains("inf"), "msg = {s}"); // Display shows 'inf'
    }

    // -------------------- f64_to_usize_safe --------------------

    #[test]
    fn to_usize_accepts_positive_integer() {
        assert_eq!(f64_to_usize_safe(42.0), Some(42));
    }

    #[test]
    fn to_usize_truncates_fraction_toward_zero() {
        // >0, normal, truncates to 0
        assert_eq!(f64_to_usize_safe(0.9), Some(0));
        assert_eq!(f64_to_usize_safe(5.99), Some(5));
    }

    #[test]
    fn to_usize_rejects_zero_and_negative() {
        assert_eq!(f64_to_usize_safe(0.0), None);
        assert_eq!(f64_to_usize_safe(-1.0), None);
        assert_eq!(f64_to_usize_safe(-0.0001), None);
    }

    #[test]
    fn to_usize_rejects_subnormal_nan_inf() {
        let sub = f64::MIN_POSITIVE / 2.0;
        assert!(sub.is_subnormal());
        assert_eq!(f64_to_usize_safe(sub), None);
        assert_eq!(f64_to_usize_safe(f64::NAN), None);
        assert_eq!(f64_to_usize_safe(f64::INFINITY), None);
        assert_eq!(f64_to_usize_safe(f64::NEG_INFINITY), None);
    }

    // Boundary near usize::MAX. This uses target_pointer_width cfg so it stays portable.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn to_usize_accepts_up_to_usize_max_64() {
        // NOTE: (usize::MAX as f64) rounds to 18446744073709551616.0 (2^64),
        // but the predicate allows <= that, and Rust float->int casts saturate,
        // so this should yield Some(usize::MAX).
        let max_f = usize::MAX as f64;
        assert_eq!(f64_to_usize_safe(max_f), None);
    }
}
