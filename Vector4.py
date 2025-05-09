from __future__ import annotations
import math
from typing import Union, Optional
import Matrix4x4
import Vector2
import Vector3
import Quaternion
from extra import staticproperty, Number

class Vector4():
    def __init__(self, x : Optional[Union[Number, Vector2.Vector2, Vector3.Vector3, Quaternion.Quaternion]] = None, y : Optional[Number] = None, z : Optional[Number] = None, w : Optional[Number] = None):
        if (isinstance(x, Quaternion.Quaternion)):
            w = x.w
            z = x.z
            y = x.y
            x = x.x 
        elif (isinstance(x, Vector3.Vector3)):
            w = w or y
            z = x.z
            y = x.y
            x = x.x
        elif (isinstance(x, Vector2.Vector2)):
            w = w or z
            z = z or y
            y = x.y
            x = x.x
        
        elif (y is None and z is None and w is None):
            y = x
            z = x
            w = x

        
        self.x : Number = x or 0
        self.y : Number = y or 0
        self.z : Number = z or 0
        self.w : Number = w or 0

    def to_string(self) -> str:
        return f"X:{self.x}, Y:{self.y}, Z:{self.z}, W:{self.w}"

    def copy_to(self, arr : list, index : int = 0):
        if (not arr):
            raise Exception("Arg_NullArgumentNullRef")
        if (index < 0 or index >= len(arr)):
            raise ValueError('Arg_ArgumentOutOfRangeException')
        if (len(arr)-index < 4):
            raise ValueError('Arg_ElementsInSourceIsGreaterThanDestination')

        arr[index] = self.x
        arr[index + 1] = self.y
        arr[index + 2] = self.z
        arr[index + 3] = self.w

    def equals(self, other : Vector4) -> bool:
        if (not isinstance(other, Vector4)):
            return False
        return self.x == other.x and \
        self.y == other.y and \
        self.z == other.z and \
        self.w == other.w

    def not_equal(self, other : Vector4) -> bool:
        if (not isinstance(other, Vector4)):
            return True
        return self.x != other.x or \
        self.y != other.y or \
        self.z != other.z or \
        self.w != other.w

    @staticmethod
    def dot(vector1 : Vector4, vector2 : Vector4) -> float: 
        return vector1.x * vector2.x + \
               vector1.y * vector2.y + \
               vector1.z * vector2.z + \
               vector1.w * vector2.w

    @staticmethod
    def min(value1 : Vector4, value2 : Vector4) -> Vector4:
        return Vector4(
            value1.x if value1.x < value2.x else value2.x, 
            value1.y if value1.y < value2.y else value2.y, 
            value1.z if value1.z < value2.z else value2.z, 
            value1.w if value1.w < value2.w else value2.w
        )

    @staticmethod
    def max(value1 : Vector4, value2 : Vector4) -> Vector4:
        return Vector4(
            value1.x if value1.x > value2.x else value2.x, 
            value1.y if value1.y > value2.y else value2.y, 
            value1.z if value1.z > value2.z else value2.z,
            value1.w if value1.w > value2.w else value2.w
        )

    @staticmethod
    def abs(value : Vector4) -> Vector4: 
        return Vector4(abs(value.x), abs(value.y), abs(value.z), abs(value.w))

    @staticmethod
    def square_root(value : Vector4) -> Vector4:
        return Vector4(math.sqrt(value.x), math.sqrt(value.y), math.sqrt(value.z), math.sqrt(value.w))

    @staticmethod
    def add(left : Vector4, right : Vector4) -> Vector4:
        return Vector4(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w)

    @staticmethod
    def subtract(left : Vector4, right : Vector4 | None = None) -> Vector4:
        if right is None: 
            right = left
            left = Vector4(0)
        return Vector4(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w)

    @staticmethod
    def multiply(left : Union[Vector4, Number], right : Union[Vector4, Number]) -> Vector4:
        if (isinstance(right, Number)):
            if (isinstance(left, Number)):
                raise ValueError('Cannot multiply 2 numbers as vectors!')
            return Vector4(left.x * right, left.y * right, left.z * right, left.w * right)
        elif (isinstance(left, Number)):
            return Vector4(right.x * left, right.y * left, right.z * left, right.w * left)
        else: 
            return Vector4(right.x * left.x, right.y * left.y, right.z * left.z, right.w * left.w)

    @staticmethod
    def divide(left : Vector4, right : Union[Vector4, Number]) -> Vector4:
        if (isinstance(right, Number)):
            invDiv = 1.0 / right
            return Vector4(left.x * invDiv, left.y * invDiv, left.z * invDiv, left.w * invDiv)
        else: 
            return Vector4(left.x / right.x, left.y / right.y, left.z / right.z, left.w / right.w)

    def __add__(self, other : Vector4) -> Vector4:
        return self.add(self, other)

    def __sub__(self, other : Vector4) -> Vector4:
        return self.subtract(self, other)

    def __mul__(self, other : Union[Vector4, Number]) -> Vector4:
        return self.multiply(self, other)

    def __rmul__(self, other : Union[Vector4, Number]) -> Vector4:
        return self.multiply(self, other)

    def __truediv__(self, other : Union[Vector4, Number]) -> Vector4:
        return self.divide(self, other)

    def __rtruediv__(self, other : Union[Vector4, Number]) -> Vector4:
        return self.divide(self, other)

    def __neg__(self) -> Vector4:
        return self.subtract(self)

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

    @staticproperty
    def zero() -> Vector4:
        return Vector4()

    @staticproperty
    def one() -> Vector4:
        return Vector4(1, 1, 1, 1)

    @staticproperty
    def unit_x() -> Vector4:
        return Vector4(1, 0, 0, 0)

    @staticproperty
    def unit_y() -> Vector4:
        return Vector4(0, 1, 0, 0)

    @staticproperty
    def unit_z() -> Vector4:
        return Vector4(0, 0, 1, 0)

    @staticproperty
    def unit_w() -> Vector4:
        return Vector4(0, 0, 0, 1)

    def length(self):
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float:
        return Vector4.dot(self, self)

    @staticmethod
    def normalize(value : Vector4) -> Vector4:
        return value / value.length()

    @staticmethod
    def distance(value1 : Vector4, value2 : Vector4) -> float: 
        return math.sqrt(Vector4.distanceSquared(value1, value2))


    @staticmethod
    def distanceSquared(value1 : Vector4, value2 : Vector4) -> float: 
        difference = value1 - value2
        return Vector4.dot(difference, difference)

    @staticmethod
    def cross(vector1 : Vector4, vector2 : Vector4) -> Vector4: 
        return Vector4 (
            vector1.y * vector2.z - vector1.z * vector2.y,
            vector1.z * vector2.x - vector1.x * vector2.z,
            vector1.x * vector2.y - vector1.y * vector2.x
            )

    @staticmethod
    def reflect(vector : Vector4, normal : Vector4):
        dot = Vector4.dot(vector, normal)
        temp = normal * dot * 2
        return vector - temp

    @staticmethod
    def clamp(value1 : Vector4, min : Vector4, max : Vector4) -> Vector4:
        x = value1.x
        x = max.x if x > max.x else x
        x = min.x if x < min.x else x

        y = value1.y
        y = max.y if y > max.y else y
        y = min.y if y < min.y else y

        z = value1.z
        z = max.z if z > max.z else z
        z = min.z if z < min.z else z

        w = value1.w
        w = max.w if w > max.w else w
        w = min.w if w < min.w else w

        return Vector4(x, y, z, w)

    @staticmethod
    def lerp(value1 : Vector4, value2 : Vector4, amount : Number) -> Vector4:
        return Vector4(
            value1.x + (value2.x - value1.x) * amount,
            value1.y + (value2.y - value1.y) * amount,
            value1.z + (value2.z - value1.z) * amount,
            value1.w + (value2.w - value1.w) * amount
            )

    @staticmethod
    def transform(position : Union[Vector2.Vector2, Vector3.Vector3, Vector4], matrix : Matrix4x4.Matrix4x4) -> Vector4:
        if isinstance(position, Vector2.Vector2):
            return Vector4(
            position.x * matrix.m11 + position.y * matrix.m21 + matrix.m41,
            position.x * matrix.m12 + position.y * matrix.m22 + matrix.m42,
            position.x * matrix.m13 + position.y * matrix.m23 + matrix.m43,
            position.x * matrix.m14 + position.y * matrix.m24 + matrix.m44
            )
        if isinstance(position, Vector3.Vector3):
            return Vector4(
            position.x * matrix.m11 + position.y * matrix.m21 + position.z * matrix.m31 + matrix.m41,
            position.x * matrix.m12 + position.y * matrix.m22 + position.z * matrix.m32 + matrix.m42,
            position.x * matrix.m13 + position.y * matrix.m23 + position.z * matrix.m33 + matrix.m43,
            position.x * matrix.m14 + position.y * matrix.m24 + position.z * matrix.m34 + matrix.m44
            )
        else:
            return Vector4(
            position.x * matrix.m11 + position.y * matrix.m21 + position.z * matrix.m31 + position.w * matrix.m41,
            position.x * matrix.m12 + position.y * matrix.m22 + position.z * matrix.m32 + position.w * matrix.m42,
            position.x * matrix.m13 + position.y * matrix.m23 + position.z * matrix.m33 + position.w * matrix.m43,
            position.x * matrix.m14 + position.y * matrix.m24 + position.z * matrix.m34 + position.w * matrix.m44
        )
    

    #--------------------
    #TODO: add quaternion method: 
    @staticmethod
    def transformQuaternion():
        pass
            
    