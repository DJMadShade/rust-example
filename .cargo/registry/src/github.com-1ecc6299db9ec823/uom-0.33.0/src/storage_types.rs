/// Macro to duplicate code on a per-storage type basis. The given code is duplicated in new
/// modules named for each storage type. A type alias, `V`, is generated that code can use for the
/// type. `@...` match arms are considered private.
///
/// * `$attr`: Module attributes. Generally used to set documentation comments for storage type
///   modules generated by the macro.
/// * `$T`: Types to generate a module for. Accepts all underlying storage types along with a number
///   of different categories:
///   * `All`: `usize`, `u8`, `u16`, `u32`, `u64`, `u128`, `isize`, `i8`, `i16`, `i32`, `i64`,
///     `i128`, `BigInt`, `BigUint`, `Rational`, `Rational32`, `Rational64`, `BigRational`,
///     `Complex32`, `Complex64`, `f32`, and `f64`.
///   * `PrimInt`: `usize`, `u8`, `u16`, `u32`, `u64`, `u128`, `isize`, `i8`, `i16`, `i32`, `i64`,
///     and `i128`.
///   * `Ratio`: `Rational`, `Rational32`, `Rational64`, and `BigRational`.
///   * `Float`: `f32` and `f64`.
///   * `Signed`: `isize`, `i8`, `i16`, `i32`, `i64`, `i128`, `BigInt`, `Rational`, `Rational32`,
///     `Rational64`, `BigRational`, `f32`, and `f64`.
///   * `Unsigned`: `usize`, `u8`, `u16`, `u32`, `u64`, `u128`, and `BigUint`.
///   * `Complex`: `Complex32` and `Complex64`.
/// * `$tt`: Code to place into each storage type module.
///
#[cfg_attr(all(feature = "f32", feature = "f64"), doc = " ```rust")]
#[cfg_attr(not(all(feature = "f32", feature = "f64")), doc = " ```rust,ignore")]
/// #[macro_use]
/// extern crate uom;
///
/// fn main() {
///     f32::do_work(1.234_f32);
///     f64::do_work(1.234_f64);
/// }
///
/// storage_types! {
///     /// Type modules.
///     pub types: Float;
///
///     pub fn do_work(_v: V) {}
/// }
/// ```
#[macro_export]
macro_rules! storage_types {
    ($(#[$attr:meta])* types: $($T:ident),+; $($tt:tt)*) => {
        storage_types!(@types ($(#[$attr])*) @mod $($T),+; ($($tt)*));
    };
    ($(#[$attr:meta])* pub types: $($T:ident),+; $($tt:tt)*) => {
        storage_types!(@types ($(#[$attr])*) @pub_mod $($T),+; ($($tt)*));
    };
    (@types $attr:tt @$M:ident $($T:ident),+; $tt:tt) => {
        $(storage_types!(@type $attr @$M $T $tt);)+
    };
    (@type ($(#[$attr:meta])*) @$M:ident usize ($($tt:tt)*)) => {
        storage_type_usize!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident u8 ($($tt:tt)*)) => {
        storage_type_u8!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident u16 ($($tt:tt)*)) => {
        storage_type_u16!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident u32 ($($tt:tt)*)) => {
        storage_type_u32!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident u64 ($($tt:tt)*)) => {
        storage_type_u64!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident u128 ($($tt:tt)*)) => {
        storage_type_u128!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident isize ($($tt:tt)*)) => {
        storage_type_isize!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident i8 ($($tt:tt)*)) => {
        storage_type_i8!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident i16 ($($tt:tt)*)) => {
        storage_type_i16!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident i32 ($($tt:tt)*)) => {
        storage_type_i32!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident i64 ($($tt:tt)*)) => {
        storage_type_i64!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident i128 ($($tt:tt)*)) => {
        storage_type_i128!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident BigInt ($($tt:tt)*)) => {
        storage_type_bigint!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident BigUint ($($tt:tt)*)) => {
        storage_type_biguint!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Rational ($($tt:tt)*)) => {
        storage_type_rational!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Rational32 ($($tt:tt)*)) => {
        storage_type_rational32!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Rational64 ($($tt:tt)*)) => {
        storage_type_rational64!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident BigRational ($($tt:tt)*)) => {
        storage_type_bigrational!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Complex32 ($($tt:tt)*)) => {
        storage_type_complex32!(($(#[$attr])*) @$M (
            /// Inner storage type.
            #[allow(dead_code)]
            pub type VV = f32;
            $($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Complex64 ($($tt:tt)*)) => {
        storage_type_complex64!(($(#[$attr])*) @$M (
            /// Inner storage type.
            #[allow(dead_code)]
            pub type VV = f64;
            $($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident f32 ($($tt:tt)*)) => {
        storage_type_f32!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident f64 ($($tt:tt)*)) => {
        storage_type_f64!(($(#[$attr])*) @$M ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident All ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M usize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u128 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M isize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i128 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigInt ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigUint ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigRational ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Complex32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Complex64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M f32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M f64 ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident PrimInt ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M usize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u128 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M isize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i128 ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Ratio ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M Rational ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigRational ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Float ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M f32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M f64 ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Signed ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M isize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M i128 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigInt ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Rational64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigRational ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M f32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M f64 ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Unsigned ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M usize ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u8 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u16 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u64 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M u128 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M BigUint ($($tt)*));
    };
    (@type ($(#[$attr:meta])*) @$M:ident Complex ($($tt:tt)*)) => {
        storage_types!(@type ($(#[$attr])*) @$M Complex32 ($($tt)*));
        storage_types!(@type ($(#[$attr])*) @$M Complex64 ($($tt)*));
    };
    (@mod ($(#[$attr:meta])*) $M:ident, $V:ty; ($($tt:tt)*)) => {
        $(#[$attr])*
        mod $M {
            /// Storage type.
            #[allow(dead_code)]
            pub type V = $V;

            $($tt)*
        }
    };
    (@pub_mod ($(#[$attr:meta])*) $M:ident, $V:ty; ($($tt:tt)*)) => {
        $(#[$attr])*
        pub mod $M {
            /// Storage type.
            #[allow(dead_code)]
            pub type V = $V;

            $($tt)*
        }
    };
    ($($tt:tt)*) => {
        storage_types! {
            types: All;

            $($tt)*
        }
    };
}

macro_rules! storage_type_types {
    ($($macro_name:ident!($feature:tt, $name:ident, $($type:tt)+);)+) => {
        $(#[macro_export]
        #[doc(hidden)]
        #[cfg(feature = $feature)]
        macro_rules! $macro_name {
            ($attr:tt @$M:ident $tt:tt) => {
                storage_types!(@$M $attr $name, $($type)+; $tt);
            };
        }

        #[macro_export]
        #[doc(hidden)]
        #[cfg(not(feature = $feature))]
        macro_rules! $macro_name {
            ($attr:tt @$M:ident $tt:tt) => {
            };
        })+
    };
}

storage_type_types! {
    storage_type_usize!("usize", usize, usize);
    storage_type_u8!("u8", u8, u8);
    storage_type_u16!("u16", u16, u16);
    storage_type_u32!("u32", u32, u32);
    storage_type_u64!("u64", u64, u64);
    storage_type_u128!("u128", u128, u128);
    storage_type_isize!("isize", isize, isize);
    storage_type_i8!("i8", i8, i8);
    storage_type_i16!("i16", i16, i16);
    storage_type_i32!("i32", i32, i32);
    storage_type_i64!("i64", i64, i64);
    storage_type_i128!("i128", i128, i128);
    storage_type_bigint!("bigint", bigint, $crate::num::BigInt);
    storage_type_biguint!("biguint", biguint, $crate::num::BigUint);
    storage_type_rational!("rational", rational, $crate::num::Rational);
    storage_type_rational32!("rational32", rational32, $crate::num::rational::Rational32);
    storage_type_rational64!("rational64", rational64, $crate::num::rational::Rational64);
    storage_type_bigrational!("bigrational", bigrational, $crate::num::BigRational);
    storage_type_complex32!("complex32", complex32, $crate::num::complex::Complex32);
    storage_type_complex64!("complex64", complex64, $crate::num::complex::Complex64);
    storage_type_f32!("f32", f32, f32);
    storage_type_f64!("f64", f64, f64);
}
