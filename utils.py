from datetime import datetime, timezone, timedelta

def get_current_datetime_cet():
    # Create a timezone object for UTC+1 (CET) without DST
    cet = timezone(timedelta(hours=1), name="CET")
    now_cet = datetime.now(cet)
    return now_cet.strftime("%Y%m%d_%H%M%S%z")