import pandas as pd


def callback_fix(df: pd.DataFrame) -> None:
    s = df["timestamp"]

    if pd.api.types.is_datetime64_any_dtype(s):
        raw = s.astype("int64")
    else:
        raw = pd.to_numeric(s, errors="coerce").astype("Int64")
        raw = raw.astype("int64")

    v = int(raw.iloc[0])

    if v < 10 ** 11:
        unit = "s"
    elif v < 10 ** 14:
        unit = "ms"
    elif v < 10 ** 17:
        unit = "us"
    else:
        unit = "ns"

    df["timestamp"] = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
