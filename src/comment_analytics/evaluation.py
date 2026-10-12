"""Measure overlap with newspaper staff, not accuracy at detecting astroturfing."""

import pandas as pd

from comment_analytics.config import FLAGS
from comment_analytics.data import require_columns, validate_ids


def staff_overlap(users: pd.DataFrame, staff: pd.DataFrame) -> dict[str, object]:
    """Return user-level staff overlap counts and nullable descriptive rates.

    Nonstaff users are unlabelled for professional status, not verified negatives.
    Rates with zero denominators are unavailable rather than reported as zero.
    """
    require_columns(staff, {"ID_User"}, "Newspaper_Staff")
    validate_ids(staff, ["ID_User"], "ID_User")
    is_staff = users["ID_User"].isin(staff["ID_User"])
    result: dict[str, object] = {
        "interpretation": "Staff overlap only; not professional-status or astroturfing accuracy.",
        "users": len(users),
        "staff_in_posts": int(is_staff.sum()),
        "staff_without_posts": int((~staff["ID_User"].isin(users["ID_User"])).sum()),
    }
    for flag in [*FLAGS, "flag_any"]:
        flagged = users[flag].astype(bool)
        overlap = int((flagged & is_staff).sum())
        result[flag] = {
            "flagged_staff": overlap,
            "flagged_nonstaff": int((flagged & ~is_staff).sum()),
            "unflagged_staff": int((~flagged & is_staff).sum()),
            "unflagged_nonstaff": int((~flagged & ~is_staff).sum()),
            "staff_share_of_flagged": overlap / int(flagged.sum()) if flagged.any() else None,
            "share_of_staff_flagged": overlap / int(is_staff.sum()) if is_staff.any() else None,
        }
    return result
