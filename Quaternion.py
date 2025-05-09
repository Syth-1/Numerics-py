from __future__ import annotations
import math
from typing import Union, Optional, overload

import Matrix4x4
import Vector2
import Vector3
import Vector4
from extra import staticproperty


from extra import Number

class Quaternion:
    def __init__(self, x : Optional[Union[Number, Vector2.Vector2, Vector3.Vector3, Vector4.Vector4]] = None, y : Optional[Number] = None, z : Optional[Number] = None, w : Optional[Number] = None):
        if (isinstance(x, Vector4.Vector4)):
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

    def equals(self, other : Quaternion): 
        if (not isinstance(other, Quaternion)):
            return False
        return self.x == other.x and \
        self.y == other.y and \
        self.z == other.z and \
        self.w == other.w

    def not_equal(self, other : Quaternion) -> bool:
        if (not isinstance(other, Quaternion)):
            return True
        return self.x != other.x or \
        self.y != other.y or \
        self.z != other.z or \
        self.w != other.w

    @staticproperty
    def identity() -> Quaternion: 
        return Quaternion(0, 0, 0, 1)

    def is_identity(self): 
        return (self.x == 0 and self.y == 0 and self.z == 0 and self.w == 1) 

    
    def length(self) -> float: 
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float: 
        return (self.x * self.x + \
               self.y * self.y + \
               self.z * self.z + \
               self.w * self.w
            )

    def normalize(self) -> Quaternion: 
        invNorm = 1 / self.length()
        return Quaternion(self.x * invNorm, self.y * invNorm, self.z * invNorm, self.w * invNorm)

    def conjugate(self) -> Quaternion:
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> Quaternion: 
        #  -1   (       a              -v       )
        # q   = ( -------------   ------------- )
        #       (  a^2 + |v|^2  ,  a^2 + |v|^2  )

        invNorm = 1 / self.length_squared()
        return Quaternion(-self.x * invNorm, -self.y * invNorm, -self.z * invNorm, self.w * invNorm)


    @staticmethod
    def create_from_axis_angle(axis : Vector3.Vector3, angle : float) -> Quaternion: 
        half_angle = angle * 0.5
        s = math.sin(half_angle)
        c = math.cos(half_angle)

        return Quaternion(axis * s, c)

    @staticmethod
    def create_from_yaw_pitch_roll(yaw : float, pitch : float, roll : float) -> Quaternion:
        #  Roll first, about axis the object is facing, then
        #  pitch upward, then yaw to face into the new heading

        half_roll  = roll * 0.5
        sr = math.sin(half_roll)
        cr = math.cos(half_roll)

        half_pitch = pitch * 0.5
        sp = math.sin(half_pitch)
        cp = math.cos(half_pitch)

        half_yaw = yaw * 0.5
        sy = math.sin(half_yaw)
        cy = math.cos(half_yaw)

        return Quaternion(
            x = cy * sp * cr + sy * cp * sr, 
            y = sy * cp * cr - cy * sp * sr,
            z = cy * cp * sr - sy * sp * cr, 
            w = cy * cp * cr + sy * sp * sr
        )

    @staticmethod
    def create_from_rotation_matrix(matrix : Matrix4x4.Matrix4x4) -> Quaternion:
        trace = matrix.m11 + matrix.m22 + matrix.m33
        q = Quaternion()

        if (trace > 0):
            s = math.sqrt(trace + 1.0)
            q.w = s * 0.5
            s = 0.5 / s
            q.x = (matrix.m23 - matrix.m32) * s
            q.y = (matrix.m31 - matrix.m13) * s
            q.z = (matrix.m12 - matrix.m21) * s

        else: 
            if (matrix.m11 >= matrix.m22 and matrix.m11 >= matrix.m33):
                s = math.sqrt(1.0 + matrix.m11 - matrix.m22 - matrix.m33)
                invS = 0.5 / s
                q.x = 0.5 * s
                q.y = (matrix.m12 + matrix.m21) * invS
                q.z = (matrix.m13 + matrix.m31) * invS
                q.w = (matrix.m23 - matrix.m32) * invS

            elif (matrix.m22 > matrix.m33):
                s = math.sqrt(1.0 + matrix.m22 - matrix.m11 - matrix.m33)
                invS = 0.5 / s
                q.x = (matrix.m21 + matrix.m12) * invS
                q.y = 0.5 * s
                q.z = (matrix.m32 + matrix.m23) * invS
                q.w = (matrix.m31 - matrix.m13) * invS

            else: 
                s = math.sqrt(1.0 + matrix.m33 - matrix.m11 - matrix.m22)
                invS = 0.5 / s
                q.x = (matrix.m31 + matrix.m13) * invS
                q.y = (matrix.m32 + matrix.m23) * invS
                q.z = 0.5 * s
                q.w = (matrix.m12 - matrix.m21) * invS

        return q


    @staticmethod
    def unity_euler_2_quaternion(pitch : Number, yaw : Number, roll : Number) -> Quaternion: 
        
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)
        roll = math.radians(roll)

        roll_over2 = roll * 0.5
        sin_roll_over2 = math.sin(roll_over2)
        cos_roll_over2 = math.cos(roll_over2)
        pitch_over2 = pitch * 0.5
        sin_pitch_over2 = math.sin(pitch_over2)
        cos_pitch_over2 = math.cos(pitch_over2)
        yaw_over2 = yaw * 0.5
        sin_yaw_over2 = math.sin(yaw_over2)
        cos_yaw_over2 = math.cos(yaw_over2)

        return Quaternion(
            cos_yaw_over2 * sin_pitch_over2 * cos_roll_over2 + sin_yaw_over2 * cos_pitch_over2 * sin_roll_over2,
            sin_yaw_over2 * cos_pitch_over2 * cos_roll_over2 - cos_yaw_over2 * sin_pitch_over2 * sin_roll_over2,
            cos_yaw_over2 * cos_pitch_over2 * sin_roll_over2 - sin_yaw_over2 * sin_pitch_over2 * cos_roll_over2,
            cos_yaw_over2 * cos_pitch_over2 * cos_roll_over2 + sin_yaw_over2 * sin_pitch_over2 * sin_roll_over2
        )
    

    @staticmethod
    def slerp(quaternion1 : Quaternion, quaternion2 : Quaternion, amount : float) -> Quaternion:
        EPSILON = 1e-6

        t = amount 

        cos_omega =  quaternion1.x * quaternion2.x + quaternion1.y * quaternion2.y + \
                    quaternion1.z * quaternion2.z + quaternion1.w * quaternion2.w
            
        flip = False

        if (cos_omega < 0.0):
            flip = True
            cos_omega = -cos_omega


        if (cos_omega > (1.0 - EPSILON)):
            # Too close, do straight linear interpolation.
            s1 = 1.0 - t
            s2 = -t if flip else t

        else:
            omega = math.acos(cos_omega)
            inv_sin_omega = (1 / math.sin(omega))

            s1 = math.sin((1.0 - t) * omega) * inv_sin_omega
            s2 = -math.sin(t * omega) * inv_sin_omega if flip else \
                  math.sin(t * omega) * inv_sin_omega

        return Quaternion(
            x = s1 * quaternion1.x + s2 * quaternion2.x, 
            y = s1 * quaternion1.y + s2 * quaternion2.y,
            z = s1 * quaternion1.z + s2 * quaternion2.z,
            w = s1 * quaternion1.w + s2 * quaternion2.w
        )

    @staticmethod
    def lerp(quaternion1 : Quaternion, quaternion2 : Quaternion, amount : float) -> Quaternion: 

        t = amount 
        t1 = 1.0 - t
        r = Quaternion() 

        dot = quaternion1.x * quaternion2.x + quaternion1.y * quaternion2.y + \
              quaternion1.z * quaternion2.z + quaternion1.w * quaternion2.w

        if (dot >= 0.0): 
            r.x = t1 * quaternion1.x + t * quaternion2.x
            r.y = t1 * quaternion1.y + t * quaternion2.y
            r.z = t1 * quaternion1.z + t * quaternion2.z
            r.w = t1 * quaternion1.w + t * quaternion2.w

        else: 
            r.x = t1 * quaternion1.x - t * quaternion2.x
            r.y = t1 * quaternion1.y - t * quaternion2.y
            r.z = t1 * quaternion1.z - t * quaternion2.z
            r.w = t1 * quaternion1.w - t * quaternion2.w

        # Normalize it.
        ls = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w
        inv_norm = 1.0 / math.sqrt(ls)

        r.x *= inv_norm
        r.y *= inv_norm
        r.z *= inv_norm
        r.w *= inv_norm

        return r

    @staticmethod
    def concatenate(value1 : Quaternion, value2 : Quaternion) -> Quaternion: 

        # Concatenate rotation is actually q2 * q1 instead of q1 * q2.
        # So that's why value2 goes q1 and value1 goes q2.
        q1x = value2.x
        q1y = value2.y
        q1z = value2.z
        q1w = value2.w

        q2x = value1.x
        q2y = value1.y
        q2z = value1.z
        q2w = value1.w

        # cross(av, bv)
        cx = q1y * q2z - q1z * q2y
        cy = q1z * q2x - q1x * q2z
        cz = q1x * q2y - q1y * q2x

        dot = q1x * q2x + q1y * q2y + q1z * q2z

        return Quaternion(
            x = q1x * q2w + q2x * q1w + cx,
            y = q1y * q2w + q2y * q1w + cy,
            z = q1z * q2w + q2z * q1w + cz,
            w = q1w * q2w - dot
        )

    @staticmethod
    def negate(value : Quaternion) -> Quaternion: 
        return Quaternion(
            x = -value.x,
            y = -value.y,
            z = -value.z,
            w = -value.w
        )

    @staticmethod
    def add(left : Quaternion, right : Quaternion) -> Quaternion: 
        return Quaternion(
            x = left.x + right.x,
            y = left.y + right.y,
            z = left.z + right.z,
            w = left.w + right.w
        )

    @staticmethod
    def subtract(left : Quaternion, right : Quaternion | None = None) -> Quaternion: 
        if right is None: 
            return Quaternion.negate(left)
        return Quaternion(
            x = left.x - right.x,
            y = left.y - right.y,
            z = left.z - right.z,
            w = left.w - right.w
        )
    
    @overload
    @staticmethod
    def multiply(left : Quaternion, right : Union[Quaternion, Number]) -> Quaternion: ...

    @overload
    @staticmethod
    def multiply(left : Quaternion, right : Vector3.Vector3) -> Vector3.Vector3: ...

    @staticmethod
    def multiply(left : Quaternion, right : Union[Quaternion, Vector3.Vector3, Number]) -> Quaternion | Vector3.Vector3: 
        if (isinstance(right, Quaternion)):
            q1x = left.x
            q1y = left.y
            q1z = left.z
            q1w = left.w

            q2x = right.x
            q2y = right.y
            q2z = right.z
            q2w = right.w

            # cross(av, bv)
            cx = q1y * q2z - q1z * q2y
            cy = q1z * q2x - q1x * q2z
            cz = q1x * q2y - q1y * q2x

            dot = q1x * q2x + q1y * q2y + q1z * q2z

            return Quaternion(
                x = q1x * q2w + q2x * q1w + cx,
                y = q1y * q2w + q2y * q1w + cy,
                z = q1z * q2w + q2z * q1w + cz,
                w = q1w * q2w - dot
            ) 
        
        #if vec3
        if (isinstance(right, Vector3.Vector3)): 
            num =   left.x * 2
            num2 =  left.y * 2
            num3 =  left.z * 2
            num4 =  left.x * num
            num5 =  left.y * num2
            num6 =  left.z * num3
            num7 =  left.x * num2
            num8 =  left.x * num3
            num9 =  left.y * num3
            num10 = left.w * num
            num11 = left.w * num2
            num12 = left.w * num3

            return Vector3.Vector3(
                (1 - (num5 + num6)) * right.x + (num7 - num12) * right.y + (num8 + num11) * right.z,
                (num7 + num12) * right.x + (1 - (num4 + num6)) * right.y + (num9 - num10) * right.z,
                (num8 - num11) * right.x + (num9 + num10) * right.y + (1 - (num4 + num5)) * right.z
            )
            

        #if number: 
        return Quaternion(
            x = left.x * right,
            y = left.y * right,
            z = left.z * right,
            w = left.w * right
        )

    @staticmethod
    def divide(left : Quaternion, right : Quaternion) -> Quaternion: 
        q1x = left.x
        q1y = left.y
        q1z = left.z
        q1w = left.w

        #-------------------------------------
        # Inverse part.
        ls =  right.x * right.x + right.y * right.y + \
              right.z * right.z + right.w * right.w

        invNorm = 1.0 / ls

        q2x = -right.x * invNorm
        q2y = -right.y * invNorm
        q2z = -right.z * invNorm
        q2w = right.w * invNorm

        #-------------------------------------
        # Multiply part.

        # cross(av, bv)
        cx = q1y * q2z - q1z * q2y
        cy = q1z * q2x - q1x * q2z
        cz = q1x * q2y - q1y * q2x

        dot = q1x * q2x + q1y * q2y + q1z * q2z

        return Quaternion(
            x = q1x * q2w + q2x * q1w + cx,
            y = q1y * q2w + q2y * q1w + cy,
            z = q1z * q2w + q2z * q1w + cz,
            w = q1w * q2w - dot
           )

    @staticmethod
    def quaternion_look_rotation(forward : Vector3.Vector3, up : Vector3.Vector3):
        forward = Vector3.Vector3.normalize(forward)

        vector = Vector3.Vector3.normalize(forward)
        vector2 = Vector3.Vector3.normalize(Vector3.Vector3.cross(up, vector))
        vector3 = Vector3.Vector3.cross(vector, vector2)
        m00 = vector2.x
        m01 = vector2.y
        m02 = vector2.z
        m10 = vector3.x
        m11 = vector3.y
        m12 = vector3.z
        m20 = vector.x
        m21 = vector.y
        m22 = vector.z


        num8 = (m00 + m11) + m22
        quaternion = Quaternion()
        if (num8 > 0):
            num = math.sqrt(num8 + 1)
            quaternion.w = num * 0.5
            num = 0.5 / num
            quaternion.x = (m12 - m21) * num
            quaternion.y = (m20 - m02) * num
            quaternion.z = (m01 - m10) * num
            return quaternion
        
        if m00 >= m11 and m00 >= m22:
            num7 = math.sqrt(((1 + m00) - m11) - m22)
            num4 = 0.5 / num7
            quaternion.x = 0.5 * num7
            quaternion.y = (m01 + m10) * num4
            quaternion.z = (m02 + m20) * num4
            quaternion.w = (m12 - m21) * num4

            return quaternion
        
        if (m11 > m22):
            num6 = math.sqrt(((1 + m11) - m00) - m22)
            num3 = 0.5 / num6
            quaternion.x = (m10+ m01) * num3
            quaternion.y = 0.5 * num6
            quaternion.z = (m21 + m12) * num3
            quaternion.w = (m20 - m02) * num3
            return quaternion; 
        
        num5 = math.sqrt(((1 + m22) - m00) - m11)
        num2 = 0.5 / num5
        quaternion.x = (m20 + m02) * num2
        quaternion.y = (m21 + m12) * num2
        quaternion.z = 0.5 * num5
        quaternion.w = (m01 - m10) * num2
        return quaternion
    

    def __add__(self, other) -> Quaternion:
        return self.add(self, other)

    def __sub__(self, other) -> Quaternion: 
        return self.subtract(self, other)

    def __mul__(self, other) -> Quaternion:
        return self.multiply(self, other)

    def __rmul__(self, other) -> Quaternion: 
        return self.multiply(self, other)

    def __neg__(self) -> Quaternion: 
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
    