import time, uuid, structlog

logger = structlog.get_logger()

def with_timing(fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("timing", function=fn.__name__, ms=round(elapsed,2))
    return wrapper

def new_trace_id():
    return str(uuid.uuid4())

def log_event(event: str, **kwargs):
    logger.info(event, **kwargs)
