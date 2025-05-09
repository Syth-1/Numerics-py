from __future__ import annotations
import copy
import math
from typing import Union, Optional

import Vector2
import Vector4
import Matrix4x4
import Quaternion
from extra import staticproperty, Number

class Vector3():
    def __init__(self, x : Optional[Union[Number, Vector2.Vector2, Vector4.Vector4, Quaternion.Quaternion]] = None, y : Optional[Number] = None, z : Optional[Number] = None):
        if isinstance(x, Quaternion.Quaternion) or isinstance(x, Vector4.Vector4):
            z = x.z
            y = x.y
            x = x.x
        elif isinstance(x, Vector2.Vector2):
            y = x.y
            x = x.x
        elif (y is None and z is None):
            y = x
            z = x

        
        self.x : Number = x or 0
        self.y : Number = y or 0
        self.z : Number = z or 0


    def to_string(self) -> str:
        return f"X:{self.x}, Y:{self.y}, Z:{self.z}"

    def copy_to(self, arr : list, index : int = 0):
        if (not arr):
            raise Exception("Arg_NullArgumentNullRef")
        if (index < 0 or index >= len(arr)):
            raise ValueError('Arg_ArgumentOutOfRangeException')
        if (len(arr)-index < 3):
            raise ValueError('Arg_ElementsInSourceIsGreaterThanDestination')

        arr[index] = self.x
        arr[index + 1] = self.y
        arr[index + 2] = self.z

    def equals(self, other : Vector3) -> bool:
        if (not isinstance(other, Vector3)):
            return False
        return self.x == other.x and \
        self.y == other.y and \
        self.z == other.z

    def not_equal(self, other : Vector3) -> bool:
        if (not isinstance(other, Vector3)):
            return True
        return self.x != other.x or \
        self.y != other.y or \
        self.z != other.z

    @staticmethod
    def dot(vector1 : Vector3, vector2 : Vector3) -> float: 
        return vector1.x * vector2.x + \
               vector1.y * vector2.y + \
               vector1.z * vector2.z

    @staticmethod
    def min(value1 : Vector3, value2 : Vector3) -> Vector3:
        return Vector3(
            value1.x if value1.x < value2.x else value2.x, 
            value1.y if value1.y < value2.y else value2.y, 
            value1.z if value1.z < value2.z else value2.z
        )

    @staticmethod
    def max(value1 : Vector3, value2 : Vector3) -> Vector3:
        return Vector3(
            value1.x if value1.x > value2.x else value2.x, 
            value1.y if value1.y > value2.y else value2.y, 
            value1.z if value1.z > value2.z else value2.z
        )

    @staticmethod
    def abs(value : Vector3) -> Vector3: 
        return Vector3(abs(value.x), abs(value.y), abs(value.z))

    @staticmethod
    def square_root(value : Vector3) -> Vector3:
        return Vector3(math.sqrt(value.x), math.sqrt(value.y), math.sqrt(value.z))

    @staticmethod
    def add(left : Vector3, right : Vector3) -> Vector3:
        return Vector3(left.x + right.x, left.y + right.y, left.z + right.z)

    @staticmethod
    def subtract(left : Vector3, right : Vector3 | None = None) -> Vector3:
        if right is None: 
            right = left
            left = Vector3(0)
        return Vector3(left.x - right.x, left.y - right.y, left.z - right.z)

    @staticmethod
    def multiply(left : Union[Vector3, Number], right : Union[Vector3, Number]) -> Vector3:
        if (isinstance(right, Number)):
            if (isinstance(left, Number)):
                raise ValueError('Cannot multiply 2 numbers as vectors!')
            return Vector3(left.x * right, left.y * right, left.z * right)
        elif (isinstance(left, Number)):
            return Vector3(right.x * left, right.y * left, right.z * left)
        else: 
            return Vector3(right.x * left.x, right.y * left.y, right.z * left.z)

    @staticmethod
    def divide(left : Vector3, right : Union[Vector3, Number]) -> Vector3:
        if (isinstance(right, Number)):
            invDiv = 1.0 / right
            return Vector3(left.x * invDiv, left.y * invDiv, left.z * invDiv)
        else: 
            return Vector3(left.x / right.x, left.y / right.y, left.z / right.z)

    def __add__(self, other : Vector3) -> Vector3:
        return self.add(self, other)

    def __sub__(self, other : Vector3) -> Vector3:
        return self.subtract(self, other)

    def __mul__(self, other : Union[Vector3, Number]) -> Vector3:
        return self.multiply(self, other)

    def __rmul__(self, other : Union[Vector3, Number]) -> Vector3:
        return self.multiply(self, other)

    def __truediv__(self, other : Union[Vector3, Number]) -> Vector3:
        return self.divide(self, other)

    def __rtruediv__(self, other : Union[Vector3, Number]) -> Vector3:
        return self.divide(self, other)

    def __neg__(self) -> Vector3:
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
    def zero() -> Vector3:
        return Vector3()

    @staticproperty
    def one() -> Vector3:
        return Vector3(1, 1, 1)

    @staticproperty
    def unit_x() -> Vector3:
        return Vector3(1, 0, 0)

    @staticproperty
    def unit_y() -> Vector3:
        return Vector3(0, 1, 0)

    @staticproperty
    def unit_z() -> Vector3:
        return Vector3(0, 0, 1)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float:
        return Vector3.dot(self, self)

    @staticmethod
    def normalize(value : Vector3) -> Vector3:
        return value / (value.length() or 1)

    @staticmethod
    def distance(value1 : Vector3, value2 : Vector3) -> float: 
        return math.sqrt(Vector3.distance_squared(value1, value2))


    @staticmethod
    def distance_squared(value1 : Vector3, value2 : Vector3) -> float: 
        difference = value1 - value2
        return Vector3.dot(difference, difference)

    @staticmethod
    def cross(vector1 : Vector3, vector2 : Vector3) -> Vector3: 
        return Vector3 (
            vector1.y * vector2.z - vector1.z * vector2.y,
            vector1.z * vector2.x - vector1.x * vector2.z,
            vector1.x * vector2.y - vector1.y * vector2.x
        )

    @staticmethod
    def reflect(vector : Vector3, normal : Vector3):
        dot = Vector3.dot(vector, normal)
        temp = normal * dot * 2
        return vector - temp

    @staticmethod
    def clamp(value1 : Vector3, min : Vector3, max : Vector3) -> Vector3:
        x = value1.x
        x = max.x if x > max.x else x
        x = min.x if x < min.x else x

        y = value1.y
        y = max.y if y > max.y else y
        y = min.y if y < min.y else y

        z = value1.z
        z = max.z if z > max.z else z
        z = min.z if z < min.z else z

        return Vector3(x, y, z)

    @staticmethod
    def lerp(value1 : Vector3, value2 : Vector3, amount : Number) -> Vector3:
        firstInfluence = value1 * (1 - amount)
        secondInfluence = value2 * amount
        return firstInfluence + secondInfluence

    def apply_matrix(self, m : Matrix4x4.Matrix4x4): 
        x = self.x
        y = self.y
        z = self.z

        w = 1 / ( m.m14 * self.x + m.m24 * self.y + m.m34 * self.z + m.m44 )

        self.x = ( m.m11 * x + m.m21 * y + m.m31 * z + m.m41 ) * w
        self.y = ( m.m12 * x + m.m22 * y + m.m32 * z + m.m42 ) * w
        self.z = ( m.m13 * x + m.m23 * y + m.m33 * z + m.m43 ) * w

        return self

    @staticmethod
    def transform(position : Vector3, matrix : Matrix4x4.Matrix4x4) -> Vector3:
        return Vector3(
            position.x * matrix.m11 + position.y * matrix.m21 + position.z * matrix.m31 + matrix.m41,
            position.x * matrix.m12 + position.y * matrix.m22 + position.z * matrix.m32 + matrix.m42,
            position.x * matrix.m13 + position.y * matrix.m23 + position.z * matrix.m33 + matrix.m43
            )
    
    @staticmethod
    def transform_normal(normal : Vector3, matrix : Matrix4x4.Matrix4x4) -> Vector3:
        return Vector3(
            normal.x * matrix.m11 + normal.y * matrix.m21 + normal.z * matrix.m31,
            normal.x * matrix.m12 + normal.y * matrix.m22 + normal.z * matrix.m32,
            normal.x * matrix.m13 + normal.y * matrix.m23 + normal.z * matrix.m33
            )

    def round(self) -> Vector3:
        c : Vector3 = copy.deepcopy(self)
        c.x = round(self.x)
        c.y = round(self.y)
        c.z = round(self.z)
        return c 


    def to_quaternion(self) -> Quaternion.Quaternion:
        return Quaternion.Quaternion.unity_euler_2_quaternion(self.x, self.y, self.z)

    @staticmethod
    def unity_quaternion_2_euler(q1 : Quaternion.Quaternion) -> Vector3: 
        sqw = q1.w * q1.w
        sqx = q1.x * q1.x
        sqy = q1.y * q1.y
        sqz = q1.z * q1.z
        unit = sqx + sqy + sqz + sqw; # if normalised is one, otherwise is correction factor
        test = q1.x * q1.w - q1.y * q1.z


        def wrap_angle(vec3 : Vector3) -> Vector3:
            def wrap(angle : Number) -> float: 
                angle = round(angle)
                while angle > 360: 
                    angle -= 360
                while angle < 0: 
                    angle += 360 
                return angle 

            vec3.x = wrap(vec3.x)
            vec3.y = wrap(vec3.y)
            vec3.z = wrap(vec3.z)

            return vec3

        RAD2DEG = 180 / math.pi

        v = Vector3()
        if (test > 0.4995 * unit): #// singularity at north pole
            v.y = 2 * math.atan2(q1.y, q1.x)
            v.x = math.pi / 2
            v.z = 0
            return wrap_angle(RAD2DEG * v)

        if (test < -0.4995 * unit): # singularity at south pole
            v.y = -2 * math.atan2(q1.y, q1.x)
            v.x = -math.pi / 2
            v.z = 0
            return wrap_angle(RAD2DEG * v)
            
        q = Quaternion.Quaternion(q1.w, q1.z, q1.x, q1.y)
        v.y = math.atan2(2 * q.x * q.w + 2 * q.y * q.z, 1 - 2 * (q.z * q.z + q.w * q.w));   # Yaw
        v.x = math.asin(2 * (q.x * q.z - q.w * q.y));                                       # Pitch
        v.z = math.atan2(2 * q.x * q.y + 2 * q.z * q.w, 1 - 2 * (q.y * q.y + q.z * q.z));   # Roll
        
        return wrap_angle(RAD2DEG * v)


    #--------------------
    #TODO: add quaternion method: 
    @staticmethod
    def transform_quaternion():
        pass
            
    @staticmethod
    def three_axis_rotation(point : Vector3, rotation : Vector3):
        return point.apply_matrix(Vector3.create_rotation_matrix(rotation))

    @staticmethod
    def create_rotation_matrix(rotation : Vector3): 
        rx = rotation.x * math.pi / 180
        ry = rotation.y * math.pi / 180
        rz = rotation.z * math.pi / 180

        return Matrix4x4.Matrix4x4(
            m11 = math.cos(rz) * math.cos(ry),
            m21 = (math.cos(rz) * math.sin(ry) * math.sin(rx)) - (math.sin(rz) * math.cos(rx)),
            m31 = (math.cos(rz) * math.sin(ry) * math.cos(rx)) + (math.sin(rz) * math.sin(rx)),
            m41 = 0,

            m12 = math.sin(rz) * math.cos(ry),
            m22 = (math.sin(rz) * math.sin(ry) * math.sin(rx)) + (math.cos(rz) * math.cos(rx)),
            m32 = (math.sin(rz) * math.sin(ry) * math.cos(rx)) - (math.cos(rz) * math.sin(rx)),
            m42 = 0,

            m13 = -math.sin(ry),
            m23 = math.cos(ry) * math.sin(rx),
            m33 = math.cos(ry) * math.cos(rx),
            m43 = 0,

            m14 = 0,
            m24 = 0,
            m34 = 0,
            m44 = 1
        )
    
    @staticmethod
    def look_at(_from : Vector3, to : Vector3):
        up = Vector3(0, 1, 0)

        pos = Vector3()
        rot = Quaternion.Quaternion(0, 0, 0, 0)
        scale = Vector3()

        view_matrix = Matrix4x4.Matrix4x4.create_look_at(_from, to, up)
        Matrix4x4.Matrix4x4.decompose(view_matrix, scale, rot, pos)
        return rot


    @staticmethod
    def three_look_at(_from : Vector3, to : Vector3):
        up = Vector3(0, 1, 0)

        view_matrix = Matrix4x4.Matrix4x4.three_create_look_at(_from, to, up)

        euler : Vector3 = Vector3.set_from_rotation_matrix(view_matrix, order="YXZ")
        return Vector3(-math.degrees(euler.x), math.degrees(euler.y) + 180 , -math.degrees(euler.z))

    @staticmethod
    def set_from_rotation_matrix( m : Matrix4x4.Matrix4x4, order : str):
        def clamp(value, min_num, max_num): 
            return min(max_num, max(min_num, value))

        x = 0 
        y = 0 
        z = 0

        match order: 
            case 'XYZ':
                y = math.asin(clamp( m.m31, - 1, 1 ) )

                if ( abs( m.m31 ) < 0.9999999 ):

                    x = math.atan2( - m.m32, m.m33 )
                    z = math.atan2( - m.m21, m.m11 )

                else:

                    x = math.atan2( m.m23, m.m22 )
                    z = 0

                
            case 'YXZ':
                x = math.asin( - clamp( m.m32, - 1, 1 ) )

                if ( abs( m.m32 ) < 0.9999999 ) :

                    y = math.atan2( m.m31, m.m33 )
                    z = math.atan2( m.m12, m.m22 )

                else :
                    y = math.atan2( - m.m13, m.m11 )
                    z = 0

            case 'ZXY':
                x = math.asin( clamp( m.m23, - 1, 1 ) )

                if ( abs( m.m23 ) < 0.9999999 ) :
                    y = math.atan2( - m.m13, m.m33 )
                    z = math.atan2( - m.m21, m.m22 )

                else :
                    y = 0
                    z = math.atan2( m.m12, m.m11 )

            case 'ZYX':
                y = math.asin( - clamp( m.m13, - 1, 1 ) )

                if ( abs( m.m13 ) < 0.9999999 ) :

                    x = math.atan2( m.m23, m.m33 )
                    z = math.atan2( m.m12, m.m11 )

                else :

                    x = 0
                    z = math.atan2( - m.m21, m.m22 ) 

            case 'YZX':
                z = math.asin( clamp( m.m12, - 1, 1 ) )

                if ( abs( m.m12 ) < 0.9999999 ) :
                    x = math.atan2( - m.m32, m.m22 )
                    y = math.atan2( - m.m13, m.m11 )

                else :
                    x = 0
                    y = math.atan2( m.m31, m.m33 )

            case 'XZY':
                z = math.asin( - clamp( m.m21, - 1, 1 ) )

                if ( abs( m.m21 ) < 0.9999999 ) :
                    x = math.atan2( m.m23, m.m22 )
                    y = math.atan2( m.m31, m.m11 )

                else :
                    x = math.atan2( - m.m32, m.m33 )
                    y = 0

            case _: 
                raise Exception(
                    f"`set_from_rotation_matrix`: encountered an unknown order: {order}"
                )
            
        return Vector3(x, y, z)