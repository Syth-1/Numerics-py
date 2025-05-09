from __future__ import annotations
import math
from typing import Union, Optional

import Matrix4x4
import Quaternion
import Vector3
import Vector4
from extra import Number

class Plane(): 
    def __init__(self, x : Union[Number, Vector3.Vector3, Vector4.Vector4],
        y : Optional[Number] = None, z : Optional[Number] = None, d : Optional[Number] = None) -> None:

        if isinstance(x, Vector4.Vector4): 
            d = x.w
            z = x.z
            y = x.y
            x = x.x
        
        elif isinstance(x, Vector3.Vector3): 
            d = d or y or 0
            z = x.z
            y = x.y
            x = x.x

        elif (y == None and z == None and d == None): 
            y = x
            z = x 
            d = x 

        y = y or 0
        z = z or 0
        d = d or 0

        self.normal : Vector3.Vector3 = Vector3.Vector3(x, y, z)
        self.d = d

    def to_string(self): 
        return f"{self.normal.to_string()}, {self.d}"

    def equals(self, other : Plane) -> bool: 
        if not isinstance(other, Plane): 
            return False

        return self.normal == other.normal and self.d == other.d
    

    def not_equal(self, other : Plane) -> bool:
        if not isinstance(other, Plane): 
            return True

        return (self.normal.x != other.normal.x or \
                self.normal.y != other.normal.y or \
                self.normal.z != other.normal.z or \
                self.d != other.d
        )

    @staticmethod
    def create_from_vertices(point1 : Vector3.Vector3, point2 : Vector3.Vector3, point3 : Vector3.Vector3) -> Plane:
        
        a = point2 - point3
        b = point3 - point1 

        n = Vector3.Vector3.cross(a, b)
        normal = Vector3.Vector3.normalize(n)

        d = -Vector3.Vector3.dot(normal, point1)

        return Plane(normal, d)

    @staticmethod
    def normalize(value : Plane) -> Plane:
        FLT_EPSILON = 1.192092896e-07 # smallest such that 1.0+FLT_EPSILON != 1.0

        normal_length_squared = value.normal.length_squared()
        if (abs(normal_length_squared - 1.0) < FLT_EPSILON): 
            # It already normalized, so we don't need to farther process.
            return value
        normal_length = math.sqrt(normal_length_squared)
        return Plane(value.normal / normal_length, value.d / normal_length)

    @staticmethod
    def transform(plane : Plane, matrix_or_quat : Union[Matrix4x4.Matrix4x4, Quaternion.Quaternion]): 
        # Compute rotation matrix.
        if isinstance(matrix_or_quat, Quaternion.Quaternion):
            x2 = matrix_or_quat.x + matrix_or_quat.x
            y2 = matrix_or_quat.y + matrix_or_quat.y
            z2 = matrix_or_quat.z + matrix_or_quat.z

            wx2 = matrix_or_quat.w * x2
            wy2 = matrix_or_quat.w * y2
            wz2 = matrix_or_quat.w * z2
            xx2 = matrix_or_quat.x * x2
            xy2 = matrix_or_quat.x * y2
            xz2 = matrix_or_quat.x * z2
            yy2 = matrix_or_quat.y * y2
            yz2 = matrix_or_quat.y * z2
            zz2 = matrix_or_quat.z * z2

            m11 = 1.0 - yy2 - zz2
            m21 = xy2 - wz2
            m31 = xz2 + wy2

            m12 = xy2 + wz2
            m22 = 1.0 - xx2 - zz2
            m32 = yz2 - wx2

            m13 = xz2 - wy2
            m23 = yz2 + wx2
            m33 = 1.0 - xx2 - yy2

            x = plane.normal.x
            y = plane.normal.y
            z = plane.normal.z

            return Plane(
                x * m11 + y * m21 + z * m31,
                x * m12 + y * m22 + z * m32,
                x * m13 + y * m23 + z * m33,
                plane.d)
        
        m = Matrix4x4.Matrix4x4()
        Matrix4x4.Matrix4x4.invert(matrix_or_quat, m)

        x = plane.normal.x
        y = plane.normal.y
        z = plane.normal.z
        w = plane.d

        return Plane(
            x * m.m11 + y * m.m12 + z * m.m13 + w * m.m14,
            x * m.m21 + y * m.m22 + z * m.m23 + w * m.m24,
            x * m.m31 + y * m.m32 + z * m.m33 + w * m.m34,
            x * m.m41 + y * m.m42 + z * m.m43 + w * m.m44)
        
    @staticmethod
    def dot(plane : Plane, value : Vector4.Vector4) -> Number: 
        return (plane.normal.x * value.x + \
                plane.normal.y * value.y + \
                plane.normal.z * value.z + \
                plane.d * value.w
            )

    @staticmethod
    def dot_coordinate(plane : Plane, value : Vector3.Vector3) -> Number: 
        return Vector3.Vector3.dot(plane.normal, value) + plane.d

    @staticmethod
    def dotNormal(plane : Plane, value : Vector3.Vector3) -> Number: 
        return Vector3.Vector3.dot(plane.normal, value)

    def __eq__(self, other) -> bool:
        return self.equals(other)
    
    def __req__(self, other) -> bool:
        return self.equals(other)

    def __ne__(self, other) -> bool:
        return self.not_equal(other)

    def __rne__(self, other) -> bool:
        return self.not_equal(other)

    def __str__(self) -> str:
        return self.to_string()