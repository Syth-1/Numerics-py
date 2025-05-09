from __future__ import annotations
import math
from typing import Union, Optional, Literal

import Matrix4x4
import Vector2
from extra import staticproperty, Matrix2String, Number

class Matrix3x2:
    def __init__(self,
                 m11: Optional[Union[Number, Matrix4x4.Matrix4x4]] = None,  # m11 is either matrix4x4, a float or None
                 m12: Optional[Number] = None,  # rest of args are either float or None
                 m21: Optional[Number] = None,
                 m22: Optional[Number] = None,
                 m31: Optional[Number] = None,
                 m32: Optional[Number] = None):
        
        # If m11 is an instance of Matrix4x4, extract the relevant parts
        if isinstance(m11, Matrix4x4.Matrix4x4):
            m32 = m11.m32
            m31 = m11.m31

            m22 = m11.m22
            m21 = m11.m21

            m12 = m11.m12
            m11 = m11.m11
        elif m12 is None and m21 is None and m22 is None and m31 is None and m32 is None:
            m12 = m11
            m21 = m11
            m22 = m11
            m31 = m11
            m32 = m11

        # Assign the values, defaulting to 0.0 where None is present
        self.m11 : Number = m11 or 0.0
        self.m12 : Number = m12 or 0.0

        self.m21 : Number = m21 or 0.0
        self.m22 : Number = m22 or 0.0

        self.m31 : Number = m31 or 0.0
        self.m32 : Number = m32 or 0.0


    @staticproperty
    def __identity(): 
        return Matrix3x2 \
        (
            1, 0,
            0, 1, 
            0, 0
        )

    def to_string(self):
        arr = [
        [self.m11, self.m12],
        [self.m21, self.m22], 
        [self.m31, self.m32]
        ]
        return(Matrix2String.to_string(arr))


    def equals(self, other : Matrix3x2) -> bool: # Check diagonal element first for early out.
        if (not isinstance(other, Matrix3x2)):
            return False
        return (self.m11 == other.m11 and self.m22 == other.m22 and 
                                                self.m12 == other.m12 and
                    self.m21 == other.m21 and
                    self.m31 == other.m31 and self.m32 == other.m32)

    def not_equal(self, other : Matrix3x2) -> bool:
        if (not isinstance(other, Matrix3x2)):
            return True
        return (
            self.m11 != other.m11 or self.m12 != other.m12 or 
            self.m22 != other.m22 or self.m21 != other.m21 or
            self.m31 != other.m31 or self.m32 != other.m32)

    @staticmethod
    def identity() -> Matrix3x2:
        return Matrix3x2.__identity
    
    @property
    def is_identity(self) -> bool: # Check diagonal element first for early out. 
        return ( 
            self.m11 == 1 and self.m22 == 1 and \
                              self.m12 == 0 and \
            self.m21 == 0 and \
            self.m31 == 0 and self.m32 == 0
        )

    @property
    def translation(self) -> Vector2.Vector2:
        return Vector2.Vector2(self.m31, self.m32)

    @translation.setter
    def translation(self, value : Vector2.Vector2): 
        self.m31 = value.x
        self.m32 = value.y


    @staticmethod   
    def create_translation(position1 : Union[Vector2.Vector2, Number], position2 : Number | None = None) -> Matrix3x2: 
        if not isinstance(position1, Vector2.Vector2):
            position1 = Vector2.Vector2(position1, position2)

        return Matrix3x2(
            m11 = 1.0,
            m12 = 0.0,
            m21 = 0.0,
            m22 = 1.0,
            m31 = position1.x,
            m32 = position1.y,
        )

    @staticmethod
    def create_scale(x_axis : Union[Vector2.Vector2, Number], y_axis : Union[Vector2.Vector2, Number, None] = None, centerPoint : Vector2.Vector2 | None = None) -> Matrix3x2:

        if isinstance(y_axis, Vector2.Vector2):
            centerPoint = y_axis
            y_axis = None

        if isinstance(x_axis, Vector2.Vector2):
            y_axis = x_axis.y
            x_axis = x_axis.x

        y_axis = y_axis or x_axis

        tx = centerPoint.x * (1 - x_axis) if centerPoint is not None else 0
        ty = centerPoint.y * (1 - y_axis) if centerPoint is not None else 0 

        return Matrix3x2(
            m11 = x_axis,
            m12 = 0.0,
            m21 = 0.0,
            m22 = y_axis,
            m31 = tx,
            m32 = ty
        )

    @staticmethod
    def create_skew(radians_x : Number, radians_y : Number, center_point : Vector2.Vector2 | None = None) -> Matrix3x2: 
        xTan = math.tan(radians_x)
        yTan = math.tan(radians_y)

        tx = -center_point.x * xTan if center_point is not None else 0
        ty = -center_point.y * yTan if center_point is not None else 0 

        return Matrix3x2(
            m11 = 1.0,
            m12 = xTan,
            m21 = yTan,
            m22 = 1.0,
            m31 = tx,
            m32 = ty
        )

    @staticmethod
    def create_rotation(radians : Number, center_point : Vector2.Vector2) -> Matrix3x2: 

        radians = math.remainder(radians, math.pi * 2)

        EPSILON = 0.001 * math.pi / 180 # 0.1% of a degree

        if (radians > -EPSILON and radians < EPSILON):
            # Exact case for zero rotation.
            c = 1
            s = 0

        elif (radians > math.pi / 2 - EPSILON and radians < math.pi / 2 + EPSILON):
            # Exact case for 90 degree rotation.
            c = 0
            s = 1

        elif (radians < -math.pi + EPSILON or radians > math.pi - EPSILON):
            # Exact case for 180 degree rotation.
            c = -1
            s = 0

        elif (radians > -math.pi / 2 - EPSILON and radians < -math.pi / 2 + EPSILON):
            # Exact case for 270 degree rotation.
            c = 0
            s = -1

        else:
            # Arbitrary rotation.
            c = math.cos(radians)
            s = math.sin(radians)


        x = center_point.x * (1 - c) + center_point.y * s if center_point is not None else 0
        y = center_point.y * (1 - c) - center_point.x * s if center_point is not None else 0

        # [  c  s ]
        # [ -s  c ]
        # [  x  y ]
        return Matrix3x2(
            m11 = c,
            m12 = s,
            m21 = -s,
            m22 = c,
            m31 = x,
            m32 = y
        )

    def get_determinant(self) -> float: 
        # There isn't actually any such thing as a determinant for a non-square matrix,
        # but this 3x2 type is really just an optimization of a 3x3 where we happen to
        # know the rightmost column is always (0, 0, 1). So we expand to 3x3 format:
        #
        #  [ M11, M12, 0 ]
        #  [ M21, M22, 0 ]
        #  [ M31, M32, 1 ]
        #
        # Sum the diagonal products:
        #  (M11 * M22 * 1) + (M12 * 0 * M31) + (0 * M21 * M32)
        #
        # Subtract the opposite diagonal products:
        #  (M31 * M22 * 0) + (M32 * 0 * M11) + (1 * M21 * M12)
        #
        # Collapse out the constants and oh look, this is just a 2x2 determinant!

        return (self.m11 * self.m22) - (self.m21 * self.m12)


    @staticmethod
    def invert(matrix : Matrix3x2, out_result : Matrix3x2) -> Literal[False] | Matrix3x2:
        det = (matrix.m11 * matrix.m22) - (matrix.m21 * matrix.m12)
        import sys
        if(abs(det) < sys.float_info.epsilon):
            return False

        invDet = 1 / det

        return Matrix3x2(
            m11 = matrix.m22 * invDet,
            m12 = -matrix.m12 * invDet,
            m21 = -matrix.m21 * invDet,
            m22 = matrix.m11 * invDet,
            m31 = (matrix.m21 * matrix.m32 - matrix.m31 * matrix.m22) * invDet,
            m32 = (matrix.m31 * matrix.m12 - matrix.m11 * matrix.m32) * invDet
        )

    @staticmethod
    def lerp(matrix1 : Matrix3x2, matrix2 : Matrix3x2, amount : Number) -> Matrix3x2:
        return Matrix3x2(
            # First column
            m11 = matrix1.m11 + (matrix2.m11 - matrix1.m11) * amount,
            m12 = matrix1.m12 + (matrix2.m12 - matrix1.m12) * amount,

            # Second column
            m21 = matrix1.m21 + (matrix2.m21 - matrix1.m21) * amount,
            m22 = matrix1.m22 + (matrix2.m22 - matrix1.m22) * amount,

            # Third column
            m31 = matrix1.m31 + (matrix2.m31 - matrix1.m31) * amount,
            m32 = matrix1.m32 + (matrix2.m32 - matrix1.m32) * amount
        )

    @staticmethod
    def negate(value : Matrix3x2) -> Matrix3x2: 
        return Matrix3x2(
            m11 = -value.m11,
            m12 = -value.m12,
            m21 = -value.m21,
            m22 = -value.m22,
            m31 = -value.m31,
            m32 = -value.m32
        )

    @staticmethod
    def add(value1 : Matrix3x2, value2 : Matrix3x2) -> Matrix3x2: 
        return Matrix3x2(
            m11 = value1.m11 + value2.m11,
            m12 = value1.m12 + value2.m12,
            m21 = value1.m21 + value2.m21,
            m22 = value1.m22 + value2.m22,
            m31 = value1.m31 + value2.m31,
            m32 = value1.m32 + value2.m32
        )

    @staticmethod
    def subtract(value1 : Matrix3x2, value2 : Matrix3x2 | None = None) -> Matrix3x2: 
        if value2 is None: 
            value2 = value1
            value1 = Matrix3x2()

        return Matrix3x2(
            m11 = value1.m11 - value2.m11,
            m12 = value1.m12 - value2.m12,
            m21 = value1.m21 - value2.m21,
            m22 = value1.m22 - value2.m22,
            m31 = value1.m31 - value2.m31,
            m32 = value1.m32 - value2.m32
        )

    @staticmethod
    def multiply(value1 : Matrix3x2, value2 : Union[Matrix3x2, Number]) -> Matrix3x2: 
        if (isinstance(value2, Matrix3x2)): 
            return Matrix3x2(
            # First column
            m11 = value1.m11 * value2.m11 + value1.m12 * value2.m21,
            m12 = value1.m11 * value2.m12 + value1.m12 * value2.m22,

            # Second column
            m21 = value1.m21 * value2.m11 + value1.m22 * value2.m21,
            m22 = value1.m21 * value2.m12 + value1.m22 * value2.m22,

            # Third column
            m31 = value1.m31 * value2.m11 + value1.m32 * value2.m21 + value2.m31,
            m32 = value1.m31 * value2.m12 + value1.m32 * value2.m22 + value2.m32
            )

        #value 2 is a number: 

        return Matrix3x2(
            m11 = value1.m11 * value2,
            m12 = value1.m12 * value2,
            m21 = value1.m21 * value2,
            m22 = value1.m22 * value2,
            m31 = value1.m31 * value2,
            m32 = value1.m32 * value2
        )

    def __add__(self, other : Matrix3x2) -> Matrix3x2:
        return self.add(self, other)

    def __sub__(self, other : Matrix3x2) -> Matrix3x2:
        return self.subtract(other, self)

    def __mul__(self, other : Union[Matrix3x2, Number]) -> Matrix3x2:
        return self.multiply(self, other)

    def __rmul__(self, other : Union[Matrix3x2, Number]) -> Matrix3x2:
        return self.multiply(self, other)

    def __neg__(self) -> Matrix3x2:
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