import warp as wp

@wp.func
def ray_triangle_intersect(
    start_pos: wp.vec3,
    end_pos: wp.vec3,
    triangle_vertex_0: wp.vec3,
    triangle_vertex_1: wp.vec3,
    triangle_vertex_2: wp.vec3,
    scale: wp.float32,
):
    """
    Returns the intersection point of a linesegment and a triangle.
    If the line does not intersect the triangle then the function returns the end position of the linesegment
    This method is based on the Möller–Trumbore intersection algorithm.
    """

    # Define epsilon for the time check
    #time_epsilon = 1e-5 * scale
    time_epsilon = 1e-6

    # Define epsilon for the parallel check
    epsilon = 1e-8

    # Get normal of the triangle
    normal = - wp.cross(triangle_vertex_1 - triangle_vertex_0, triangle_vertex_2 - triangle_vertex_0) # flip normal
    normal = normal / wp.length(normal)

    # Calculate the edges of the triangle
    edge1 = triangle_vertex_1 - triangle_vertex_0 
    edge2 = triangle_vertex_2 - triangle_vertex_0

    # Get direction of the line segment
    direction = end_pos - start_pos

    # Calculate the determinant
    h = wp.cross(direction, edge2)
    a = wp.dot(edge1, h)

    # Check if the ray is parallel to the triangle
    if a > -epsilon and a < epsilon:
        return end_pos, 1.0, 1.0

    # Calculate the inverse of the determinant
    f = 1.0 / a
    s = start_pos - triangle_vertex_0
    u = f * wp.dot(s, h)

    # Check if the intersection point is inside the triangle
    if u < 0.0 or u > 1.0:
        return end_pos, 1.0, 1.0

    # Calculate the second determinant
    q = wp.cross(s, edge1)
    v = f * wp.dot(direction, q)

    # Check if the intersection point is inside the triangle
    if v < 0.0 or u + v > 1.0:
        return end_pos, 1.0, 1.0

    # Calculate the distance to the intersection point
    t = f * wp.dot(edge2, q)

    # Check if direction is in the same direction as the normal
    if wp.dot(normal, direction) >= 0.0:
        return end_pos, 1.0, 1.0

    # Check if the intersection point is behind the start position
    if t < time_epsilon:
        return start_pos, 0.0, wp.abs(t)
    elif t < 1.0:
        return start_pos + t * direction, t, t
    else:
        return end_pos, 1.0, 1.0

    #if t < 0.0:
    #    return end_pos, 1.0
    #elif t < time_epsilon:
    #    return start_pos, 0.0
    #elif t < (1.0 - time_epsilon):
    #    return start_pos + t * direction, t
    #elif t < 1.0 + time_epsilon:
    #    return start_pos + (t - 2.0 * time_epsilon) * direction, (t - 2.0 * time_epsilon)
    #else:
    #    return end_pos, 1.0
