import hashlib
from typing import Union

UserId = Union[int, str]

def user_splitter(
    user_id: UserId,
    n_buckets: int = 3,
    salt: str = "v1",
) -> int:
    
    key = f"{salt}:{user_id}"
    h = hashlib.md5(key.encode()).hexdigest()
    
    return int(h, 16) % n_buckets