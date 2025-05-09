from __future__ import annotations
import math
from typing import Union, Optional
import Quaternion
import Vector3
import Vector4
import Matrix4x4
from extra import staticproperty, Number

class Vector2:
    def __init__(self, x : Optional[Union[Number, Vector3.Vector3, Vector4.Vector4]] = None, y : Optional[Number] = None):
        if isinstance(x, Quaternion.Quaternion) or isinstance(x, Vector4.Vector4) or isinstance(x, Vector3.Vector3):
            y = x.y
            x = x.x
        elif y is None:
            y = x 

        self.x : Number = x or 0
        self.y : Number = y or 0

    def to_string(self) -> str:
        return f"X:{self.x}, Y:{self.y}"

    def copy_to(self, arr : list, index : int = 0):
        if (not arr):
            raise Exception("Arg_NullArgumentNullRef")
        if (index < 0 or index >= len(arr)):
            raise ValueError('Arg_ArgumentOutOfRangeException')
        if (len(arr)-index < 2):
            raise ValueError('Arg_ElementsInSourceIsGreaterThanDestination')

        arr[index] = self.x
        arr[index + 1] = self.y

    def equals(self, other : Vector2) -> bool:
        if (not isinstance(other, Vector2)):
            return False
        return self.x == other.x and \
        self.y == other.y

    def not_equal(self, other : Vector2) -> bool:
        if (not isinstance(other, Vector2)):
            return True
        return self.x != other.x or \
        self.y != other.y

    @staticmethod
    def dot(vector1 : Vector2, vector2 : Vector2) -> float: 
        return vector1.x * vector2.x + \
               vector1.y * vector2.y

    @staticmethod
    def min(value1 : Vector2, value2 : Vector2) -> Vector2:
        return Vector2(
            value1.x if value1.x < value2.x else value2.x, 
            value1.y if value1.y < value2.y else value2.y
        )

    @staticmethod
    def max(value1 : Vector2, value2 : Vector2) -> Vector2:
        return Vector2(
            value1.x if value1.x > value2.x else value2.x, 
            value1.y if value1.y > value2.y else value2.y
        )

    @staticmethod
    def abs(value : Vector2) -> Vector2: 
        return Vector2(abs(value.x), abs(value.y))

    @staticmethod
    def square_root(value : Vector2) -> Vector2:
        return Vector2(math.sqrt(value.x), math.sqrt(value.y))

    @staticmethod
    def add(left : Vector2, right : Vector2) -> Vector2:
        return Vector2(left.x + right.x, left.y + right.y)

    @staticmethod
    def subtract(left : Vector2, right : Vector2 | None = None) -> Vector2:
        if right is None: 
            right = left
            left = Vector2(0)
        return Vector2(left.x - right.x, left.y - right.y)

    @staticmethod
    def multiply(left : Union[Vector2, Number], right : Union[Vector2, Number]) -> Vector2:
        if (isinstance(right, Number)):
            if (isinstance(left, Number)):
                raise ValueError('Cannot multiply 2 numbers as vectors!')
            return Vector2(left.x * right, left.y * right)
        elif (isinstance(left, Number)):
            return Vector2(right.x * left, right.y * left)
        else: 
            return Vector2(right.x * left.x, right.y * left.y)

    @staticmethod
    def divide(left : Vector2, right : Union[Vector2, Number]) -> Vector2:
        if (isinstance(right, Number)):
            invDiv = 1.0 / right
            return Vector2(left.x * invDiv, left.y * invDiv)
        else: 
            return Vector2(left.x / right.x, left.y / right.y)

    def __add__(self, other : Vector2) -> Vector2:
        return self.add(self, other)

    def __sub__(self, other : Vector2) -> Vector2:
        return self.subtract(self, other)

    def __mul__(self, other : Union[Vector2, Number]) -> Vector2:
        return self.multiply(self, other)

    def __rmul__(self, other : Union[Vector2, Number]) -> Vector2:
        return self.multiply(self, other)

    def __truediv__(self, other : Union[Vector2, Number]) -> Vector2:
        return self.divide(self, other)

    def __rtruediv__(self, other : Vector2) -> Vector2:
        print(other)
        return self.divide(other, self)

    def __neg__(self) -> Vector2:
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
    def zero() -> Vector2:
        return Vector2()

    @staticproperty
    def one() -> Vector2:
        return Vector2(1, 1)

    @staticproperty
    def unit_x() -> Vector2:
        return Vector2(1, 0)

    @staticproperty
    def unit_y() -> Vector2:
        return Vector2(0, 1)

    def length(self):
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float:
        return Vector2.dot(self, self)

    @staticmethod
    def normalize(value : Vector2) -> Vector2:
        return value / value.length()

    @staticmethod
    def distance(value1 : Vector2, value2 : Vector2) -> float: 
        return math.sqrt(Vector2.distance_squared(value1, value2))


    @staticmethod
    def distance_squared(value1 : Vector2, value2 : Vector2) -> float: 
        difference = value1 - value2
        return Vector2.dot(difference, difference)


    @staticmethod
    def reflect(vector : Vector2, normal : Vector2):
        dot = Vector2.dot(vector, normal)
        temp = normal * dot * 2
        return vector - temp

    @staticmethod
    def clamp(value1 : Vector2, min : Vector2, max : Vector2) -> Vector2:
        x = value1.x
        x = max.x if x > max.x else x
        x = min.x if x < min.x else x

        y = value1.y
        y = max.y if y > max.y else y
        y = min.y if y < min.y else y

        return Vector2(x, y)

    @staticmethod
    def lerp(value1 : Vector2, value2 : Vector2, amount : Number) -> Vector2:
        return Vector2(
            value1.x + (value2.x - value1.x) * amount,
            value1.y + (value2.y - value1.y) * amount
            )

    @staticmethod
    def transform(position : Vector2, matrix : Matrix4x4.Matrix4x4) -> Vector2:
        return Vector2(
            position.x * matrix.m11 + position.y * matrix.m21 + matrix.m41,
            position.x * matrix.m12 + position.y * matrix.m22 + matrix.m42
            )
    
    @staticmethod
    def transformNormal(normal : Vector2, matrix : Matrix4x4.Matrix4x4) -> Vector2:
        return Vector2(
            normal.x * matrix.m11 + normal.y * matrix.m21,
            normal.x * matrix.m12 + normal.y * matrix.m22
            )

    #--------------------
    #TODO: add quaternion method: 
    @staticmethod
    def transformQuaternion():
        pass