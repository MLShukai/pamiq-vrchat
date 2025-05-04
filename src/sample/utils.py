type size_2d = int | tuple[int, int]


def size_2d_to_int_tuple(size: size_2d) -> tuple[int, int]:
    """Convert `size_2d` type to int tuple."""
    return (size, size) if isinstance(size, int) else (size[0], size[1])
