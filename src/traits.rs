#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Zero,
    Replicate,
    Reflect,
    Symmetric,
    Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    None,
    Same,
    Minimal,
    NextFastLen,
    Explicit2(usize, usize),
    Explicit3(usize, usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    Independent,
    LumaOnly,
    IgnoreAlpha,
    PremultipliedAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangePolicy {
    PreserveInput,
    Clamp01,
    ClampNegPos1,
    Unbounded,
}
