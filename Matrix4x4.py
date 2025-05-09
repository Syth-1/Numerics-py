from __future__ import annotations
import math
from typing import Union, Optional

import Quaternion
import Vector3
import Matrix3x2
import Plane
from extra import Matrix2String, staticproperty, Number
class Matrix4x4():
    def __init__(self,
        m11 : Optional[Union[Number, Matrix3x2.Matrix3x2]] = None, m12 : Optional[Number] = None, m13 : Optional[Number] = None, m14 : Optional[Number] = None, 
        m21 : Optional[Number] = None, m22 : Optional[Number] = None, m23 : Optional[Number] = None, m24 : Optional[Number] = None, 
        m31 : Optional[Number] = None, m32 : Optional[Number] = None, m33 : Optional[Number] = None, m34 : Optional[Number] = None,
        m41 : Optional[Number] = None, m42 : Optional[Number] = None, m43 : Optional[Number] = None, m44 : Optional[Number] = None,
    ):

        if (isinstance(m11, Matrix3x2.Matrix3x2)):
            m32 = m11.m32
            m31 = m11.m31

            m22 = m11.m22
            m21 = m11.m21

            m12 = m11.m12
            m11 = m11.m11

            m33 = 1
            m44 = 1
        elif (m12 == None and m13 == None and m14 == None and \
            m21 == None and m22 == None and m23 == None and m24 == None and \
            m31 == None and m32 == None and m33 == None and m34 == None and \
            m41 == None and m42 == None and m43 == None and m44 == None):

            m12 = m11
            m13 = m11
            m14 = m11
            m21 = m11
            m22 = m11
            m23 = m11
            m24 = m11
            m31 = m11
            m32 = m11
            m33 = m11
            m34 = m11
            m41 = m11
            m42 = m11
            m43 = m11
            m44 = m11

        self.m11 : Number = m11 or 0.0
        self.m12 : Number = m12 or 0.0
        self.m13 : Number = m13 or 0.0
        self.m14 : Number = m14 or 0.0

        self.m21 : Number = m21 or 0.0
        self.m22 : Number = m22 or 0.0
        self.m23 : Number = m23 or 0.0
        self.m24 : Number = m24 or 0.0

        self.m31 : Number = m31 or 0.0
        self.m32 : Number = m32 or 0.0
        self.m33 : Number = m33 or 0.0
        self.m34 : Number = m34 or 0.0

        self.m41 : Number = m41 or 0.0
        self.m42 : Number = m42 or 0.0
        self.m43 : Number = m43 or 0.0
        self.m44 : Number = m44 or 0.0


    def to_string(self):
        arr = [
        [self.m11, self.m12, self.m13, self.m14],
        [self.m21, self.m22, self.m23, self.m24], 
        [self.m31, self.m32, self.m33, self.m34],
        [self.m41, self.m42, self.m43, self.m44]
        ]
        return(Matrix2String.to_string(arr))


    def equals(self, other : Matrix4x4) -> bool: # Check diagonal element first for early out.
        if (not isinstance(other, Matrix4x4)):
            return False
        return (self.m11 == other.m11 and self.m22 == other.m22 and self.m33 == other.m33 and self.m44 == other.m44 and
                                          self.m12 == other.m12 and self.m13 == other.m13 and self.m14 == other.m14 and
                    self.m21 == other.m21 and self.m23 == other.m23 and self.m24 == other.m24 and 
                    self.m31 == other.m31 and self.m32 == other.m32 and self.m34 == other.m34 and 
                    self.m41 == other.m41 and self.m42 == other.m42 and self.m43 == other.m43)


    def not_equal(self, other : Matrix4x4) -> bool:
        if (not isinstance(other, Matrix4x4)):
            return True
        return (
            self.m11 != other.m11 or self.m12 != other.m12 or self.m13 != other.m13 or self.m14 != other.m14 or
            self.m21 != other.m21 or self.m22 != other.m22 or self.m23 != other.m23 or self.m24 != other.m24 or
            self.m31 != other.m31 or self.m32 != other.m32 or self.m33 != other.m33 or self.m34 != other.m34 or
            self.m41 != other.m41 or self.m42 != other.m42 or self.m43 != other.m43 or self.m44 != other.m44)


    @staticproperty
    def __identity() -> Matrix4x4:
        return Matrix4x4 \
        (
            1, 0, 0, 0, 
            0, 1, 0, 0,
            0, 0, 1, 0, 
            0, 0, 0, 1
        )

    @staticmethod
    def identity() -> Matrix4x4:
        return Matrix4x4.__identity

    def is_identity(self) -> bool: # Check diagonal element first for early out.
        return self.m11 == 1 and self.m22 == 1 and self.m33 == 1 and self.m44 == 1 and \
        self.m12 == 0 and self.m13 == 0 and self.m14 == 0 and \
        self.m21 == 0 and self.m23 == 0 and self.m24 == 0 and \
        self.m31 == 0 and self.m32 == 0 and self.m34 == 0 and \
        self.m41 == 0 and self.m42 == 0 and self.m43 == 0


    @property
    def translation(self) -> Vector3.Vector3: 
        return Vector3.Vector3(self.m41, self.m42, self.m43)

    @translation.setter
    def translation(self, value : Vector3.Vector3): 
        self.m41 = value.x
        self.m42 = value.y
        self.m43 = value.z

    @staticmethod
    def create_bill_board(object_position : Vector3.Vector3, camera_position : Vector3.Vector3, \
        camera_up_vector : Vector3.Vector3, camera_forward_vector : Vector3.Vector3) -> Matrix4x4:

        EPSILON = 1e-4

        z_axis = Vector3.Vector3 (
            object_position.x - camera_position.x,
            object_position.y - camera_position.y,
            object_position.z - camera_position.z
        )

        norm = z_axis.length_squared()

        if (norm < EPSILON): 
            z_axis = -camera_forward_vector
        else: 
            z_axis = Vector3.Vector3.multiply(z_axis, 1.0 / math.sqrt(norm))

        x_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(camera_up_vector, z_axis))

        y_axis = Vector3.Vector3.cross(z_axis, x_axis)

        return Matrix4x4(
            m11 = x_axis.x,
            m12 = x_axis.y,
            m13 = x_axis.z,
            m14 = 0.0,
            m21 = y_axis.x,
            m22 = y_axis.y,
            m23 = y_axis.z,
            m24 = 0.0,
            m31 = z_axis.x,
            m32 = z_axis.y,
            m33 = z_axis.z,
            m34 = 0.0,
            m41 = object_position.x,
            m42 = object_position.y,
            m43 = object_position.z,
            m44 = 1.0
        )

    @staticmethod
    def create_constrained_billboard(object_position : Vector3.Vector3, camera_position : Vector3.Vector3, \
        rotate_axis : Vector3.Vector3, camera_forward_vector : Vector3.Vector3, object_forward_vector : Vector3.Vector3) -> Matrix4x4:

        EPSILON = 1e-4
        MIN_ANGLE =1 - (0.1 * math.pi / 180) # 0.1 degrees

        face_dir = Vector3.Vector3(
            object_position.x - camera_position.x,
            object_position.y - camera_position.y,
            object_position.z - camera_position.z
        )

        norm = face_dir.length_squared(); 

        if (norm < EPSILON):
            face_dir = -camera_forward_vector 
        else: 
            face_dir = Vector3.Vector3.multiply(face_dir, (1.0 / math.sqrt(norm)))

        y_axis = rotate_axis
        x_axis = Vector3.Vector3()
        z_axis = Vector3.Vector3() 

        dot = Vector3.Vector3.dot(rotate_axis, face_dir)


        if (abs(dot) > MIN_ANGLE):

            z_axis = object_forward_vector

            # Make sure passed values are useful for compute.
            dot = Vector3.Vector3.dot(rotate_axis, z_axis)

            if (abs(dot) > MIN_ANGLE):
                z_axis = Vector3.Vector3(0, 0, 1) if (abs(rotate_axis.z) > MIN_ANGLE) else Vector3.Vector3(0, 0, -1)

            x_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(rotate_axis, z_axis))
            z_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(x_axis, rotate_axis))
        else:
            x_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(rotate_axis, face_dir))
            z_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(x_axis, y_axis))

        return Matrix4x4(
            m11 = x_axis.x,
            m12 = x_axis.y,
            m13 = x_axis.z,
            m14 = 0.0,
            m21 = y_axis.x,
            m22 = y_axis.y,
            m23 = y_axis.z,
            m24 = 0.0,
            m31 = z_axis.x,
            m32 = z_axis.y,
            m33 = z_axis.z,
            m34 = 0.0,
            m41 = object_position.x,
            m42 = object_position.y,
            m43 = object_position.z,
            m44 = 1.0
        )

    @staticmethod
    def createTranslation(x_position : Union[Number, Vector3.Vector3], y_position : Number | None = None, z_position : Number | None = None) -> Matrix4x4: 
        if isinstance(x_position, Vector3.Vector3): 
            z_position = x_position.z
            y_position = x_position.y
            x_position = x_position.x

        return Matrix4x4(
            m11 = 1.0,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,
            m21 = 0.0,
            m22 = 1.0,
            m23 = 0.0,
            m24 = 0.0,
            m31 = 0.0,
            m32 = 0.0,
            m33 = 1.0,
            m34 = 0.0,
            m41 = x_position,
            m42 = y_position,
            m43 = z_position,
            m44 = 1.0
        )

    @staticmethod
    def createScale(x_scale : Union[Vector3.Vector3, Number], y_scale : Union[Vector3.Vector3, Number, None] = None, 
        z_scale : Number | None = None, center_point : Vector3.Vector3 | None = None) -> Matrix4x4:

        if isinstance(y_scale, Vector3.Vector3):
            center_point = y_scale
            y_scale = None

        if isinstance(x_scale, Vector3.Vector3):
            z_scale = x_scale.z
            y_scale = x_scale.y
            x_scale = x_scale.x

        y_scale = y_scale or x_scale
        z_scale = z_scale or x_scale

        t_x = center_point.x * (1 - x_scale) if center_point is not None else 0
        t_y = center_point.y * (1 - y_scale) if center_point is not None else 0 
        t_z = center_point.z * (1 - z_scale) if center_point is not None else 0 

        return Matrix4x4(
            m11 = x_scale,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,
            m21 = 0.0,
            m22 = y_scale,
            m23 = 0.0,
            m24 = 0.0,
            m31 = 0.0,
            m32 = 0.0,
            m33 = z_scale,
            m34 = 0.0,
            m41 = t_x,
            m42 = t_y,
            m43 = t_z,
            m44 = 1.0,
        )

    @staticmethod
    def create_rotation_x(radians : Number, center_point : Vector3.Vector3 | None = None) -> Matrix4x4:

        c = math.cos(radians)
        s = math.sin(radians)

        y = center_point.y * (1 - c) + center_point.z * s if center_point else 0
        z = center_point.z * (1 - c) - center_point.y * s if center_point else 0

        # [  1  0  0  0 ]
        # [  0  c  s  0 ]
        # [  0 -s  c  0 ]
        # [  0  0  0  1 ]
        return Matrix4x4(
            m11 = 1.0,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,
            m21 = 0.0,
            m22 = c,
            m23 = s,
            m24 = 0.0,
            m31 = 0.0,
            m32 = -s,
            m33 = c,
            m34 = 0.0,
            m41 = 0.0,
            m42 = y,
            m43 = z,
            m44 = 1.0
        )

    @staticmethod
    def create_rotation_y(radians : Number, center_point : Vector3.Vector3 | None = None) -> Matrix4x4: 
        
        c = math.cos(radians)
        s = math.sin(radians)

        x = center_point.x * (1 - c) - center_point.z * s if center_point else 0
        z = center_point.z * (1 - c) + center_point.x * s if center_point else 0 

        # [  c  0 -s  0 ]
        # [  0  1  0  0 ]
        # [  s  0  c  0 ]
        # [  x  0  z  1 ]
        return Matrix4x4(
            m11 = c,
            m12 = 0.0,
            m13 = -s,
            m14 = 0.0,
            m21 = 0.0,
            m22 = 1.0,
            m23 = 0.0,
            m24 = 0.0,
            m31 = s,
            m32 = 0.0,
            m33 = c,
            m34 = 0.0,
            m41 = x,
            m42 = 0.0,
            m43 = z,
            m44 = 1.0
        )

    @staticmethod
    def create_rotation_z(radians : Number, center_point : Vector3.Vector3 | None = None) -> Matrix4x4: 

        c = math.cos(radians)
        s = math.sin(radians)

        x = center_point.x * (1 - c) + center_point.y * s if center_point else 0
        y = center_point.y * (1 - c) - center_point.x * s if center_point else 0

        # [  c  s  0  0 ]
        # [ -s  c  0  0 ]
        # [  0  0  1  0 ]
        # [  x  y  0  1 ]
        return Matrix4x4(
            m11 = c,
            m12 = s,
            m13 = 0.0,
            m14 = 0.0,
            m21 = -s,
            m22 = c,
            m23 = 0.0,
            m24 = 0.0,
            m31 = 0.0,
            m32 = 0.0,
            m33 = 1.0,
            m34 = 0.0,
            m41 = x,
            m42 = y,
            m43 = 0.0,
            m44 = 1.0
        )
    
    @staticmethod
    def create_from_axis_angle(axis : Vector3.Vector3, angle : Number) -> Matrix4x4: 

        # a: angle
        # x, y, z: unit vector for axis.
        #
        # Rotation matrix M can compute by using below equation.
        #
        #        T               T
        #  M = uu + (cos a)( I-uu ) + (sin a)S
        #
        # Where:
        #
        #  u = ( x, y, z )
        #
        #      [  0 -z  y ]
        #  S = [  z  0 -x ]
        #      [ -y  x  0 ]
        #
        #      [ 1 0 0 ]
        #  I = [ 0 1 0 ]
        #      [ 0 0 1 ]
        #
        #
        #     [  xx+cosa*(1-xx)   yx-cosa*yx-sina*z zx-cosa*xz+sina*y ]
        # M = [ xy-cosa*yx+sina*z    yy+cosa(1-yy)  yz-cosa*yz-sina*x ]
        #     [ zx-cosa*zx-sina*y zy-cosa*zy+sina*x   zz+cosa*(1-zz)  ]
        #

        x = axis.x
        y = axis.y
        z = axis.z

        sa = math.sin(angle)
        ca = math.cos(angle)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z

        return Matrix4x4(
            m11 = xx + ca * (1.0 - xx),
            m12 = xy - ca * xy + sa * z,
            m13 = xz - ca * xz - sa * y,
            m14 = 0.0,
            m21 = xy - ca * xy - sa * z,
            m22 = yy + ca * (1.0 - yy),
            m23 = yz - ca * yz + sa * x,
            m24 = 0.0,
            m31 = xz - ca * xz + sa * y,
            m32 = yz - ca * yz - sa * x,
            m33 = zz + ca * (1.0 - zz),
            m34 = 0.0,
            m41 = 0.0,
            m42 = 0.0,
            m43 = 0.0,
            m44 = 1.0
        )

    @staticmethod
    def createPerspectiveFieldOfView(field_of_view : Number, aspect_ratio : Number, near_plane_distance : Number, far_plane_distance : Number) -> Matrix4x4: 

        if (field_of_view <= 0 or field_of_view >= math.pi): 
            raise Exception("fieldOfView out of range!")
        
        if (near_plane_distance <= 0.0):
            raise Exception("nearPlaneDistance out of range!")

        if (far_plane_distance <= 0.0):
            raise Exception("farPlaneDistance out of range!")

        if (near_plane_distance >= far_plane_distance): 
            raise Exception("nearPlaneDistance out of range!")

        y_scale = 1.0 / math.tan(field_of_view * 0.5)
        x_scale = y_scale / aspect_ratio

        return Matrix4x4(
            m11 = x_scale,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,

            m21 = 0.0,
            m22 = y_scale,
            m23 = 0.0,
            m24 = 0/0,

            m31 = 0.0,
            m32 = 0.0,
            m33 = far_plane_distance / (near_plane_distance - far_plane_distance),
            m34 = -1.0,

            m41 = 0.0,
            m42 = 0.0,
            m43 = near_plane_distance * far_plane_distance / (near_plane_distance - far_plane_distance),
            m44 = 0.0
        )

    @staticmethod
    def create_perspective(width : Number, height : Number, near_plane_distance : Number, far_plane_distance : Number) -> Matrix4x4: 
        if (near_plane_distance <= 0.0): 
            raise Exception("nearPlaneDistance out of range!")

        if (far_plane_distance <= 0.0):
            raise Exception("farPlaneDistance out of range!")

        if (near_plane_distance >= far_plane_distance):
            raise Exception("nearPlaneDistance out of range!")

        return Matrix4x4(
            m11 = 2.0 * near_plane_distance / width,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,

            m21 = 0.0,
            m22 = 2.0 * near_plane_distance / height,
            m23 = 0.0,
            m24 = 0.0,

            m31 = 0.0,
            m32 = 0.0,
            m33 = far_plane_distance / (near_plane_distance - far_plane_distance),
            m34 = -1.0,

            m41 = 0.0,
            m42 = 0.0,
            m43 = near_plane_distance * far_plane_distance / (near_plane_distance - far_plane_distance),
            m44 = 0.0
        )

    @staticmethod
    def create_perspectiv_off_center(left : Number, right : Number, bottom : Number, top : Number,
        near_plane_distance : Number, far_plane_distance : Number) -> Matrix4x4: 

        if (near_plane_distance <= 0.0):
            raise Exception("nearPlaneDistance out of range!")

        if (far_plane_distance <= 0.0): 
            raise Exception("farPlaneDistance out of range!")

        if (near_plane_distance >= far_plane_distance):
            raise Exception("nearPlaneDistance out of range!")

        return Matrix4x4(
            m11 = 2.0 * near_plane_distance / (right - left),
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,

            m21 = 0.0,
            m22 = 2.0 * near_plane_distance / (top - bottom),
            m23 = 0.0,
            m24 = 0.0,

            m31 = (left + right) / (right - left),
            m32 = (top + bottom) / (top - bottom),
            m33 = far_plane_distance / (near_plane_distance - far_plane_distance),
            m34 = -1.0,

            m41 = 0.0,
            m42 = 0.0,
            m43 = near_plane_distance * far_plane_distance / (near_plane_distance - far_plane_distance),
            m44 = 0.0
        )

    @staticmethod
    def create_orthographic(width : Number, height : Number, z_near_plane : Number, z_far_plane : Number) -> Matrix4x4: 
        return Matrix4x4(
            m11 = 2.0 / width,
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,

            m21 = 0.0,
            m22 = 2.0 / height,
            m23 = 0.0,
            m24 = 0.0,

            m31 = 0.0,
            m32 = 0.0,
            m33 = 1.0 / (z_near_plane - z_far_plane),
            m34 = 0.0,

            m41 = 0.0,
            m42 = 0.0,
            m43 = z_near_plane / (z_near_plane - z_far_plane),
            m44 = 1.0
        )

    @staticmethod
    def create_orthographic_off_center(left : Number, right : Number, bottom : Number, top : Number,
        z_near_plane : Number, z_far_plane : Number) -> Matrix4x4: 

        return Matrix4x4(
            m11 = 2.0 / (right - left),
            m12 = 0.0,
            m13 = 0.0,
            m14 = 0.0,

            m21 = 0.0,
            m22 = 2.0 / (top - bottom),
            m23 = 0.0,
            m24 = 0.0,

            m31 = 0.0,
            m32 = 0.0,
            m33 = 1.0 / (z_near_plane - z_far_plane),
            m34 = 0.0,

            m41 = (left + right) / (left - right),
            m42 = (top + bottom) / (bottom - top),
            m43 = z_near_plane / (z_near_plane - z_far_plane),
            m44 = 1.0
        )

    @staticmethod
    def create_alt_look_at(eye : Vector3.Vector3, to : Vector3.Vector3,
        up : Vector3.Vector3) -> Matrix4x4: 

        z_axis = Vector3.Vector3.normalize(eye - to)
        x_axis = Vector3.Vector3.cross(z_axis, Vector3.Vector3.normalize(up))
        y_axis = Vector3.Vector3.cross(x_axis, z_axis)

        return Matrix4x4(
            m11 = x_axis.x,
            m21 = y_axis.y,
            m31 = z_axis.z,
            m41 = -Vector3.Vector3.dot(x_axis, eye),

            m12 = x_axis.y,
            m22 = y_axis.y,
            m32 = z_axis.y,
            m42 = -Vector3.Vector3.dot(y_axis, eye),

            m13 = x_axis.z,
            m23 = y_axis.z,
            m33 = z_axis.z,
            m43 = -Vector3.Vector3.dot(z_axis, eye),

            m14 = 0,
            m24 = 0,
            m34 = 0,
            m44 = 1.0
        )

    @staticmethod
    def create_look_at(camera_position : Vector3.Vector3, camera_target : Vector3.Vector3,
        camera_up_vector : Vector3.Vector3) -> Matrix4x4: 

        z_axis = Vector3.Vector3.normalize(camera_position - camera_target)
        x_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(camera_up_vector, z_axis))
        y_axis = Vector3.Vector3.cross(z_axis, x_axis)

        return Matrix4x4(
             m11 = x_axis.x,
             m12 = y_axis.x,
             m13 = z_axis.x,
             m14 = 0.0,
             m21 = x_axis.y,
             m22 = y_axis.y,
             m23 = z_axis.y,
             m24 = 0.0,
             m31 = x_axis.z,
             m32 = y_axis.z,
             m33 = z_axis.z,
             m34 = 0.0,
             m41 = -Vector3.Vector3.dot(x_axis, camera_position),
             m42 = -Vector3.Vector3.dot(y_axis, camera_position),
             m43 = -Vector3.Vector3.dot(z_axis, camera_position),
             m44 = 1.0
        )

    @staticmethod
    def create_world(position : Vector3.Vector3, forward : Vector3.Vector3, up : Vector3.Vector3) -> Matrix4x4: 

        z_axis = Vector3.Vector3.normalize(-forward)
        x_axis = Vector3.Vector3.normalize(Vector3.Vector3.cross(up, z_axis))
        y_axis = Vector3.Vector3.cross(z_axis, x_axis)

        return Matrix4x4(
            m11 = x_axis.x,
            m12 = x_axis.y,
            m13 = x_axis.z,
            m14 = 0.0,
            m21 = y_axis.x,
            m22 = y_axis.y,
            m23 = y_axis.z,
            m24 = 0.0,
            m31 = z_axis.x,
            m32 = z_axis.y,
            m33 = z_axis.z,
            m34 = 0.0,
            m41 = position.x,
            m42 = position.y,
            m43 = position.z,
            m44 = 1.0
        )

    @staticmethod
    def create_from_quaternion(quaternion : Quaternion.Quaternion) -> Matrix4x4:
        x_x = quaternion.x * quaternion.x
        y_y = quaternion.y * quaternion.y
        z_z = quaternion.z * quaternion.z

        x_y = quaternion.x * quaternion.y
        w_z = quaternion.z * quaternion.w
        x_z = quaternion.z * quaternion.x
        w_y = quaternion.y * quaternion.w
        y_z = quaternion.y * quaternion.z
        w_x = quaternion.x * quaternion.w

        return Matrix4x4(
            m11 = 1.0 - 2.0 * (y_y + z_z),
            m12 = 2.0 * (x_y + w_z),
            m13 = 2.0 * (x_z - w_y),
            m14 = 0.0,
            m21 = 2.0 * (x_y - w_z),
            m22 = 1.0 - 2.0 * (z_z + x_x),
            m23 = 2.0 * (y_z + w_x),
            m24 = 0.0,
            m31 = 2.0 * (x_z + w_y),
            m32 = 2.0 * (y_z - w_x),
            m33 = 1.0 - 2.0 * (y_y + x_x),
            m34 = 0.0,
            m41 = 0.0,
            m42 = 0.0,
            m43 = 0.0,
            m44 = 1.0
        )

    @staticmethod
    def create_from_yaw_pitch_roll(yaw : Number, pitch : Number, roll : Number) -> Matrix4x4: 
        q = Quaternion.Quaternion.create_from_yaw_pitch_roll(yaw, pitch, roll)
        return Matrix4x4.create_from_quaternion(q)

    @staticmethod
    def create_rotation_matrix(rotation : Vector3.Vector3) -> Matrix4x4:

        rotation *= (math.pi / 180) #convert to radians

        mat4x4 = Matrix4x4()

        mat4x4.m11 = math.cos(rotation.z) * math.cos(rotation.y)
        mat4x4.m12 = (math.cos(rotation.z) * math.sin(rotation.y) * math.sin(rotation.x)) - (math.sin(rotation.z) * math.cos(rotation.x))
        mat4x4.m13 = (math.cos(rotation.z) * math.sin(rotation.y) * math.cos(rotation.x)) + (math.sin(rotation.z) * math.sin(rotation.x))

        mat4x4.m21 = math.sin(rotation.z) * math.cos(rotation.y)
        mat4x4.m22 = (math.sin(rotation.z) * math.sin(rotation.y) * math.sin(rotation.x)) + (math.cos(rotation.z) * math.cos(rotation.x))
        mat4x4.m23 = (math.sin(rotation.z) * math.sin(rotation.y) * math.cos(rotation.x)) - (math.cos(rotation.z) * math.sin(rotation.x))

        mat4x4.m31 = -math.sin(rotation.y)
        mat4x4.m32 = math.cos(rotation.y) * math.sin(rotation.x)
        mat4x4.m33 = math.cos(rotation.y) * math.cos(rotation.x)

        mat4x4.m44 = 1

        return mat4x4

    @staticmethod
    def create_shadow(light_direction : Vector3.Vector3, plane : Plane.Plane) -> Matrix4x4: 
        p = Plane.Plane.normalize(plane)

        dot = p.normal.x * light_direction.x + p.normal.y * light_direction.y + p.normal.z * light_direction.z
        a = -p.normal.x
        b = -p.normal.y
        c = -p.normal.z
        d = -p.d

        return Matrix4x4(
            m11 = a * light_direction.x + dot,
            m21 = b * light_direction.x,
            m31 = c * light_direction.x,
            m41 = d * light_direction.x,
            m12 = a * light_direction.y,
            m22 = b * light_direction.y + dot,
            m32 = c * light_direction.y,
            m42 = d * light_direction.y,
            m13 = a * light_direction.z,
            m23 = b * light_direction.z,
            m33 = c * light_direction.z + dot,
            m43 = d * light_direction.z,

            m14 = 0.0,
            m24 = 0.0,
            m34 = 0.0,
            m44 = dot
        )

    @staticmethod
    def create_reflection(value : Plane.Plane) -> Matrix4x4:
        value = Plane.Plane.normalize(value)

        a = value.normal.x
        b = value.normal.y
        c = value.normal.z

        fa = -2.0 * a
        fb = -2.0 * b
        fc = -2.0 * c

        return Matrix4x4(
            m11 = fa * a + 1.0,
            m12 = fb * a,
            m13 = fc * a,
            m14 = 0.0,

            m21 = fa * b,
            m22 = fb * b + 1.0,
            m23 = fc * b,
            m24 = 0.0,

            m31 = fa * c,
            m32 = fb * c,
            m33 = fc * c + 1.0,
            m34 = 0.0,

            m41 = fa * value.d,
            m42 = fb * value.d,
            m43 = fc * value.d,
            m44 = 1.0
        )

    def get_determinant(self) -> Number:
        # | a b c d |     | f g h |     | e g h |     | e f h |     | e f g |
        # | e f g h | = a | j k l | - b | i k l | + c | i j l | - d | i j k |
        # | i j k l |     | n o p |     | m o p |     | m n p |     | m n o |
        # | m n o p |
        #
        #   | f g h |
        # a | j k l | = a ( f ( kp - lo ) - g ( jp - ln ) + h ( jo - kn ) )
        #   | n o p |
        #
        #   | e g h |     
        # b | i k l | = b ( e ( kp - lo ) - g ( ip - lm ) + h ( io - km ) )
        #   | m o p |     
        #
        #   | e f h |
        # c | i j l | = c ( e ( jp - ln ) - f ( ip - lm ) + h ( in - jm ) )
        #   | m n p |
        #
        #   | e f g |
        # d | i j k | = d ( e ( jo - kn ) - f ( io - km ) + g ( in - jm ) )
        #   | m n o |
        #
        # Cost of operation
        # 17 adds and 28 muls.
        #
        # add: 6 + 8 + 3 = 17
        # mul: 12 + 16 = 28

        a = self.m11
        b = self.m12
        c = self.m13
        d = self.m14

        e = self.m21
        f = self.m22
        g = self.m23
        h = self.m24

        i = self.m31
        j = self.m32
        k = self.m33
        l = self.m34

        m = self.m41
        n = self.m42
        o = self.m43
        p = self.m44

        kp_lo = k * p - l * o
        jp_ln = j * p - l * n
        jo_kn = j * o - k * n
        ip_lm = i * p - l * m
        io_km = i * o - k * m
        in_jm = i * n - j * m

        return (
            a * (f * kp_lo - g * jp_ln + h * jo_kn) - \
            b * (e * kp_lo - g * ip_lm + h * io_km) + \
            c * (e * jp_ln - f * ip_lm + h * in_jm) - \
            d * (e * jo_kn - f * io_km + g * in_jm)
        )
    @staticmethod
    def invert(matrix : Matrix4x4, result : Matrix4x4): 

        #                                       -1
        # If you have matrix M, inverse Matrix M   can compute
        #
        #     -1       1      
        #    M   = --------- A
        #            det(M)
        #
        # A is adjugate (adjoint) of M, where,
        #
        #      T
        # A = C
        #
        # C is Cofactor matrix of M, where,
        #           i + j
        # C   = (-1)      * det(M  )
        #  ij                    ij
        #
        #     [ a b c d ]
        # M = [ e f g h ]
        #     [ i j k l ]
        #     [ m n o p ]
        #
        # First Row
        #           2 | f g h |
        # C   = (-1)  | j k l | = + ( f ( kp - lo ) - g ( jp - ln ) + h ( jo - kn ) )
        #  11         | n o p |
        #
        #           3 | e g h |
        # C   = (-1)  | i k l | = - ( e ( kp - lo ) - g ( ip - lm ) + h ( io - km ) )
        #  12         | m o p |
        #
        #           4 | e f h |
        # C   = (-1)  | i j l | = + ( e ( jp - ln ) - f ( ip - lm ) + h ( in - jm ) )
        #  13         | m n p |
        #
        #           5 | e f g |
        # C   = (-1)  | i j k | = - ( e ( jo - kn ) - f ( io - km ) + g ( in - jm ) )
        #  14         | m n o |
        #
        # Second Row
        #           3 | b c d |
        # C   = (-1)  | j k l | = - ( b ( kp - lo ) - c ( jp - ln ) + d ( jo - kn ) )
        #  21         | n o p |
        #
        #           4 | a c d |
        # C   = (-1)  | i k l | = + ( a ( kp - lo ) - c ( ip - lm ) + d ( io - km ) )
        #  22         | m o p |
        #
        #           5 | a b d |
        # C   = (-1)  | i j l | = - ( a ( jp - ln ) - b ( ip - lm ) + d ( in - jm ) )
        #  23         | m n p |
        #
        #           6 | a b c |
        # C   = (-1)  | i j k | = + ( a ( jo - kn ) - b ( io - km ) + c ( in - jm ) )
        #  24         | m n o |
        #
        # Third Row
        #           4 | b c d |
        # C   = (-1)  | f g h | = + ( b ( gp - ho ) - c ( fp - hn ) + d ( fo - gn ) )
        #  31         | n o p |
        #
        #           5 | a c d |
        # C   = (-1)  | e g h | = - ( a ( gp - ho ) - c ( ep - hm ) + d ( eo - gm ) )
        #  32         | m o p |
        #
        #           6 | a b d |
        # C   = (-1)  | e f h | = + ( a ( fp - hn ) - b ( ep - hm ) + d ( en - fm ) )
        #  33         | m n p |
        #
        #           7 | a b c |
        # C   = (-1)  | e f g | = - ( a ( fo - gn ) - b ( eo - gm ) + c ( en - fm ) )
        #  34         | m n o |
        #
        # Fourth Row
        #           5 | b c d |
        # C   = (-1)  | f g h | = - ( b ( gl - hk ) - c ( fl - hj ) + d ( fk - gj ) )
        #  41         | j k l |
        #
        #           6 | a c d |
        # C   = (-1)  | e g h | = + ( a ( gl - hk ) - c ( el - hi ) + d ( ek - gi ) )
        #  42         | i k l |
        #
        #           7 | a b d |
        # C   = (-1)  | e f h | = - ( a ( fl - hj ) - b ( el - hi ) + d ( ej - fi ) )
        #  43         | i j l |
        #
        #           8 | a b c |
        # C   = (-1)  | e f g | = + ( a ( fk - gj ) - b ( ek - gi ) + c ( ej - fi ) )
        #  44         | i j k |
        #
        # Cost of operation
        # 53 adds, 104 muls, and 1 div.


        a = matrix.m11
        b = matrix.m12
        c = matrix.m13
        d = matrix.m14
        e = matrix.m21
        f = matrix.m22
        g = matrix.m23
        h = matrix.m24
        i = matrix.m31
        j = matrix.m32
        k = matrix.m33
        l = matrix.m34
        m = matrix.m41
        n = matrix.m42
        o = matrix.m43
        p = matrix.m44

        kp_lo = k * p - l * o
        jp_ln = j * p - l * n
        jo_kn = j * o - k * n
        ip_lm = i * p - l * m
        io_km = i * o - k * m
        in_jm = i * n - j * m

        a11 = +(f * kp_lo - g * jp_ln + h * jo_kn)
        a12 = -(e * kp_lo - g * ip_lm + h * io_km)
        a13 = +(e * jp_ln - f * ip_lm + h * in_jm)
        a14 = -(e * jo_kn - f * io_km + g * in_jm)

        det = a * a11 + b * a12 + c * a13 + d * a14
        import sys
        if (abs(det) < sys.float_info.epsilon):
            result = Matrix4x4()
            return False

        invDet = 1.0 / det

        result.m11 = a11 * invDet
        result.m21 = a12 * invDet
        result.m31 = a13 * invDet
        result.m41 = a14 * invDet

        result.m12 = -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet
        result.m22 = +(a * kp_lo - c * ip_lm + d * io_km) * invDet
        result.m32 = -(a * jp_ln - b * ip_lm + d * in_jm) * invDet
        result.m42 = +(a * jo_kn - b * io_km + c * in_jm) * invDet

        gp_ho = g * p - h * o
        fp_hn = f * p - h * n
        fo_gn = f * o - g * n
        ep_hm = e * p - h * m
        eo_gm = e * o - g * m
        en_fm = e * n - f * m

        result.m13 = +(b * gp_ho - c * fp_hn + d * fo_gn) * invDet
        result.m23 = -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet
        result.m33 = +(a * fp_hn - b * ep_hm + d * en_fm) * invDet
        result.m43 = -(a * fo_gn - b * eo_gm + c * en_fm) * invDet

        gl_hk = g * l - h * k
        fl_hj = f * l - h * j
        fk_gj = f * k - g * j
        el_hi = e * l - h * i
        ek_gi = e * k - g * i
        ej_fi = e * j - f * i

        result.m14 = -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet
        result.m24 = +(a * gl_hk - c * el_hi + d * ek_gi) * invDet
        result.m34 = -(a * fl_hj - b * el_hi + d * ej_fi) * invDet
        result.m44 = +(a * fk_gj - b * ek_gi + c * ej_fi) * invDet

        return True



    @staticmethod
    def decompose(matrix : Matrix4x4, scale : Vector3.Vector3, rotation : Quaternion.Quaternion,
        translation : Vector3.Vector3) -> bool:

        translation_internal = Vector3.Vector3() 
        rotation_internal = Quaternion.Quaternion()
        scale_internal = Vector3.Vector3()

        result = True 
        EPSILON = 0.0001

        def pvec_2_mat(p_vector_basis : list[Vector3.Vector3]) -> Matrix4x4: #call before getting val from matTemp
            return Matrix4x4(m11 = p_vector_basis[0].x, 
                             m12 = p_vector_basis[0].y, 
                             m13 = p_vector_basis[0].z, 
                             m14 = 0.0, 
                             m21 = p_vector_basis[1].x, 
                             m22 = p_vector_basis[1].y, 
                             m23 = p_vector_basis[1].z, 
                             m24 = 0.0, 
                             m31 = p_vector_basis[2].x, 
                             m32 = p_vector_basis[2].y, 
                             m33 = p_vector_basis[2].z, 
                             m34 = 0.0, 
                             m41 = 0.0, 
                             m42 = 0.0, 
                             m43 = 0.0, 
                             m44 = 1.0)

        def set_scale(pf_scale):
            return Vector3.Vector3(pf_scale[0], pf_scale[1], pf_scale[2])

        p_vector_basis = [Vector3.Vector3(matrix.m11, matrix.m12, matrix.m13), 
                        Vector3.Vector3(matrix.m21, matrix.m22, matrix.m23),
                        Vector3.Vector3(matrix.m31, matrix.m32, matrix.m33)]

        mat_temp = pvec_2_mat(p_vector_basis)

        p_canonical_basis = [Vector3.Vector3.unit_x, Vector3.Vector3.unit_y, Vector3.Vector3.unit_z]

        translation_internal = Vector3.Vector3(
            matrix.m41,
            matrix.m42,
            matrix.m43)

        scale_internal.x = p_vector_basis[0].length()
        scale_internal.y = p_vector_basis[1].length()
        scale_internal.z = p_vector_basis[2].length()

        a = 0
        b = 0
        c = 0

        pf_scales = [scale_internal.x, scale_internal.y, scale_internal.z]

        x = pf_scales[0]
        y = pf_scales[1]
        z = pf_scales[2]

        if (x < y):
            if (y < z):
                a = 2
                b = 1
                c = 0
            else:
                a = 1

                if (x < z):
                    b = 2
                    c = 0
                else:
                    b = 0
                    c = 2
        else:
            if (x < z):
                a = 2
                b = 0
                c = 1
            else:
                a = 0

                if (y < z):
                    b = 2
                    c = 1
                else:
                    b = 1
                    c = 2


        if (pf_scales[a] < EPSILON):
            p_vector_basis[a] = p_canonical_basis[a]

        p_vector_basis[a] = Vector3.Vector3.normalize(p_vector_basis[a])

        if (pf_scales[b] < EPSILON):
            cc = 0

            fAbsX = abs(p_vector_basis[a].x)
            fAbsY = abs(p_vector_basis[a].y)
            fAbsZ = abs(p_vector_basis[a].z)

            ## region Ranking
            if (fAbsX < fAbsY):
                if (fAbsY < fAbsZ):
                    cc = 0
                else:
                    if (fAbsX < fAbsZ):
                        cc = 0
                    else:
                        cc = 2
            else:
                if (fAbsX < fAbsZ):
                    cc = 1
                else:
                    if (fAbsY < fAbsZ):
                        cc = 1
                    else:
                        cc = 2

            p_vector_basis[b] = Vector3.Vector3.cross(p_vector_basis[a], p_canonical_basis[cc])

        p_vector_basis[b] = Vector3.Vector3.normalize(p_vector_basis[b])

        if (pf_scales[c] < EPSILON):
            p_vector_basis[c] = Vector3.Vector3.cross(p_vector_basis[a], p_vector_basis[b])

        p_vector_basis[c] = Vector3.Vector3.normalize(p_vector_basis[c])

        mat_temp = pvec_2_mat(p_vector_basis)
        det = mat_temp.get_determinant()

        # use Kramer's rule to check for handedness of coordinate system
        if (det < 0.0):
            # switch coordinate system by negating the scale and inverting the basis vector on the x-axis
            pf_scales[a] = -pf_scales[a]
            scale_internal = set_scale(pf_scales)
            p_vector_basis[a] = -(p_vector_basis[a])

            det = -det

        det -= 1.0
        det *= det

        if ((EPSILON < det)):
            # Non-SRT matrix encountered
            rotation_internal = Quaternion.Quaternion.identity
            result = False
        else:
            # generate the quaternion from the matrix
            mat_temp = pvec_2_mat(p_vector_basis)
            rotation_internal = Quaternion.Quaternion.create_from_rotation_matrix(mat_temp)


        #set with ref, not override the val! 
        scale.x = scale_internal.x
        scale.y = scale_internal.y
        scale.z = scale_internal.z

        rotation.x = rotation_internal.x
        rotation.y = rotation_internal.y
        rotation.z = rotation_internal.z
        rotation.w = rotation_internal.w

        translation.x = translation_internal.x
        translation.y = translation_internal.y
        translation.z = translation_internal.z

        return result

    
    @staticmethod
    def transform(value : Matrix4x4, rotation : Quaternion.Quaternion) -> Matrix4x4: 
        #Compute rotation matrix.
        x2 = rotation.x + rotation.x
        y2 = rotation.y + rotation.y
        z2 = rotation.z + rotation.z

        wx2 = rotation.w * x2
        wy2 = rotation.w * y2
        wz2 = rotation.w * z2
        xx2 = rotation.x * x2
        xy2 = rotation.x * y2
        xz2 = rotation.x * z2
        yy2 = rotation.y * y2
        yz2 = rotation.y * z2
        zz2 = rotation.z * z2

        q11 = 1.0 - yy2 - zz2
        q21 = xy2 - wz2
        q31 = xz2 + wy2

        q12 = xy2 + wz2
        q22 = 1.0 - xx2 - zz2
        q32 = yz2 - wx2

        q13 = xz2 - wy2
        q23 = yz2 + wx2
        q33 = 1.0 - xx2 - yy2

        return Matrix4x4( 
            m11 = value.m11 * q11 + value.m12 * q21 + value.m13 * q31,
            m12 = value.m11 * q12 + value.m12 * q22 + value.m13 * q32,
            m13 = value.m11 * q13 + value.m12 * q23 + value.m13 * q33,
            m14 = value.m14,

            #second row
            m21 = value.m21 * q11 + value.m22 * q21 + value.m23 * q31,
            m22 = value.m21 * q12 + value.m22 * q22 + value.m23 * q32,
            m23 = value.m21 * q13 + value.m22 * q23 + value.m23 * q33,
            m24 = value.m24,

            #third row
            m31 = value.m31 * q11 + value.m32 * q21 + value.m33 * q31,
            m32 = value.m31 * q12 + value.m32 * q22 + value.m33 * q32,
            m33 = value.m31 * q13 + value.m32 * q23 + value.m33 * q33,
            m34 = value.m34,

            #fourth row
            m41 = value.m41 * q11 + value.m42 * q21 + value.m43 * q31,
            m42 = value.m41 * q12 + value.m42 * q22 + value.m43 * q32,
            m43 = value.m41 * q13 + value.m42 * q23 + value.m43 * q33,
            m44 = value.m44
        )

    @staticmethod
    def transpose(matrix : Matrix4x4) -> Matrix4x4: 
        return Matrix4x4(
            m11 = matrix.m11,
            m12 = matrix.m21,
            m13 = matrix.m31,
            m14 = matrix.m41,
            m21 = matrix.m12,
            m22 = matrix.m22,
            m23 = matrix.m32,
            m24 = matrix.m42,
            m31 = matrix.m13,
            m32 = matrix.m23,
            m33 = matrix.m33,
            m34 = matrix.m43,
            m41 = matrix.m14,
            m42 = matrix.m24,
            m43 = matrix.m34,
            m44 = matrix.m44
        )

    @staticmethod
    def lerp(matrix1 : Matrix4x4, matrix2 : Matrix4x4, amount : Number) -> Matrix4x4:
        return Matrix4x4( 
            #first row
            m11 =  matrix1.m11 + (matrix2.m11 - matrix1.m11) * amount,
            m12 =  matrix1.m12 + (matrix2.m12 - matrix1.m12) * amount,
            m13 =  matrix1.m13 + (matrix2.m13 - matrix1.m13) * amount,
            m14 =  matrix1.m14 + (matrix2.m14 - matrix1.m14) * amount,
            #second row
            m21 = matrix1.m21 + (matrix2.m21 - matrix1.m21) * amount,
            m22 = matrix1.m22 + (matrix2.m22 - matrix1.m22) * amount,
            m23 = matrix1.m23 + (matrix2.m23 - matrix1.m23) * amount,
            m24 = matrix1.m24 + (matrix2.m24 - matrix1.m24) * amount,
            #third row
            m31 = matrix1.m31 + (matrix2.m31 - matrix1.m31) * amount,
            m32 = matrix1.m32 + (matrix2.m32 - matrix1.m32) * amount,
            m33 = matrix1.m33 + (matrix2.m33 - matrix1.m33) * amount,
            m34 = matrix1.m34 + (matrix2.m34 - matrix1.m34) * amount,
            #fourth row
            m41 = matrix1.m41 + (matrix2.m41 - matrix1.m41) * amount,
            m42 = matrix1.m42 + (matrix2.m42 - matrix1.m42) * amount,
            m43 = matrix1.m43 + (matrix2.m43 - matrix1.m43) * amount,
            m44 = matrix1.m44 + (matrix2.m44 - matrix1.m44) * amount
        )

    @staticmethod
    def negate(value : Matrix4x4): 
        return Matrix4x4(
            m11 = -value.m11,
            m12 = -value.m12,
            m13 = -value.m13,
            m14 = -value.m14,
            m21 = -value.m21,
            m22 = -value.m22,
            m23 = -value.m23,
            m24 = -value.m24,
            m31 = -value.m31,
            m32 = -value.m32,
            m33 = -value.m33,
            m34 = -value.m34,
            m41 = -value.m41,
            m42 = -value.m42,
            m43 = -value.m43,
            m44 = -value.m44
        )

    @staticmethod
    def add(value1 : Matrix4x4, value2 : Matrix4x4) -> Matrix4x4: 
        return Matrix4x4(
            m11 = value1.m11 + value2.m11,
            m12 = value1.m12 + value2.m12,
            m13 = value1.m13 + value2.m13,
            m14 = value1.m14 + value2.m14,
            m21 = value1.m21 + value2.m21,
            m22 = value1.m22 + value2.m22,
            m23 = value1.m23 + value2.m23,
            m24 = value1.m24 + value2.m24,
            m31 = value1.m31 + value2.m31,
            m32 = value1.m32 + value2.m32,
            m33 = value1.m33 + value2.m33,
            m34 = value1.m34 + value2.m34,
            m41 = value1.m41 + value2.m41,
            m42 = value1.m42 + value2.m42,
            m43 = value1.m43 + value2.m43,
            m44 = value1.m44 + value2.m44
        )

    @staticmethod
    def subtract(value1 : Matrix4x4, value2 : Matrix4x4) -> Matrix4x4: 
        return Matrix4x4(
            m11 = value1.m11 - value2.m11,
            m12 = value1.m12 - value2.m12,
            m13 = value1.m13 - value2.m13,
            m14 = value1.m14 - value2.m14,
            m21 = value1.m21 - value2.m21,
            m22 = value1.m22 - value2.m22,
            m23 = value1.m23 - value2.m23,
            m24 = value1.m24 - value2.m24,
            m31 = value1.m31 - value2.m31,
            m32 = value1.m32 - value2.m32,
            m33 = value1.m33 - value2.m33,
            m34 = value1.m34 - value2.m34,
            m41 = value1.m41 - value2.m41,
            m42 = value1.m42 - value2.m42,
            m43 = value1.m43 - value2.m43,
            m44 = value1.m44 - value2.m44
        )

    @staticmethod
    def multiply(value1 : Matrix4x4, value2 : Union[Matrix4x4, Number]) -> Matrix4x4: 
        if isinstance(value2, Number): 
            return Matrix4x4(
                m11 = value1.m11 * value2,
                m12 = value1.m12 * value2,
                m13 = value1.m13 * value2,
                m14 = value1.m14 * value2,
                m21 = value1.m21 * value2,
                m22 = value1.m22 * value2,
                m23 = value1.m23 * value2,
                m24 = value1.m24 * value2,
                m31 = value1.m31 * value2,
                m32 = value1.m32 * value2,
                m33 = value1.m33 * value2,
                m34 = value1.m34 * value2,
                m41 = value1.m41 * value2,
                m42 = value1.m42 * value2,
                m43 = value1.m43 * value2,
                m44 = value1.m44 * value2
            )

        return Matrix4x4( 
            #first row
            m11 = value1.m11 * value2.m11 + value1.m12 * value2.m21 + value1.m13 * value2.m31 + value1.m14 * value2.m41,
            m12 = value1.m11 * value2.m12 + value1.m12 * value2.m22 + value1.m13 * value2.m32 + value1.m14 * value2.m42,
            m13 = value1.m11 * value2.m13 + value1.m12 * value2.m23 + value1.m13 * value2.m33 + value1.m14 * value2.m43,
            m14 = value1.m11 * value2.m14 + value1.m12 * value2.m24 + value1.m13 * value2.m34 + value1.m14 * value2.m44,
            
            #second row
            m21 = value1.m21 * value2.m11 + value1.m22 * value2.m21 + value1.m23 * value2.m31 + value1.m24 * value2.m41,
            m22 = value1.m21 * value2.m12 + value1.m22 * value2.m22 + value1.m23 * value2.m32 + value1.m24 * value2.m42,
            m23 = value1.m21 * value2.m13 + value1.m22 * value2.m23 + value1.m23 * value2.m33 + value1.m24 * value2.m43,
            m24 = value1.m21 * value2.m14 + value1.m22 * value2.m24 + value1.m23 * value2.m34 + value1.m24 * value2.m44,

            #third row
            m31 = value1.m31 * value2.m11 + value1.m32 * value2.m21 + value1.m33 * value2.m31 + value1.m34 * value2.m41,
            m32 = value1.m31 * value2.m12 + value1.m32 * value2.m22 + value1.m33 * value2.m32 + value1.m34 * value2.m42,
            m33 = value1.m31 * value2.m13 + value1.m32 * value2.m23 + value1.m33 * value2.m33 + value1.m34 * value2.m43,
            m34 = value1.m31 * value2.m14 + value1.m32 * value2.m24 + value1.m33 * value2.m34 + value1.m34 * value2.m44,

            #fourth row
            m41 = value1.m41 * value2.m11 + value1.m42 * value2.m21 + value1.m43 * value2.m31 + value1.m44 * value2.m41,
            m42 = value1.m41 * value2.m12 + value1.m42 * value2.m22 + value1.m43 * value2.m32 + value1.m44 * value2.m42,
            m43 = value1.m41 * value2.m13 + value1.m42 * value2.m23 + value1.m43 * value2.m33 + value1.m44 * value2.m43,
            m44 = value1.m41 * value2.m14 + value1.m42 * value2.m24 + value1.m43 * value2.m34 + value1.m44 * value2.m44
        )

    def __add__(self, other : Matrix4x4) -> Matrix4x4:
        return self.add(self, other)

    def __sub__(self, other : Matrix4x4) -> Matrix4x4:
        return self.subtract(other, self)

    def __mul__(self, other : Union[Matrix4x4, Number]) -> Matrix4x4:
        return self.multiply(self, other)

    def __rmul__(self, other : Union[Matrix4x4, Number]) -> Matrix4x4:
        return self.multiply(self, other)

    def __neg__(self) -> Matrix4x4:
        return self.negate(self)

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

    # three.js helper functions

    def toArray(self): 
        return [
            self.m11,
            self.m12,
            self.m13,
            self.m14,

            self.m21,
            self.m22,
            self.m23,
            self.m24,

            self.m31,
            self.m32,
            self.m33,
            self.m34,

            self.m41,
            self.m42,
            self.m43,
            self.m44
        ]
    
    def set(self, n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 ):

        self.m11 = n11; self.m21 = n12; self.m31 = n13; self.m41 = n14
        self.m12 = n21; self.m22 = n22; self.m32 = n23; self.m42 = n24
        self.m13 = n31; self.m23 = n32; self.m33 = n33; self.m43 = n34
        self.m14 = n41; self.m24 = n42; self.m44 = n43; self.m44 = n44

        return self
    
    @staticmethod
    def multiply_matrix(mat1 : Matrix4x4, mat2 : Matrix4x4): 
        a11 = mat1.m11
        a12 = mat1.m21
        a13 = mat1.m31
        a14 = mat1.m41
        a21 = mat1.m12
        a22 = mat1.m22
        a23 = mat1.m32
        a24 = mat1.m42
        a31 = mat1.m13
        a32 = mat1.m23
        a33 = mat1.m33
        a34 = mat1.m43
        a41 = mat1.m14
        a42 = mat1.m24
        a43 = mat1.m34
        a44 = mat1.m44

        b11 = mat2.m11
        b12 = mat2.m21
        b13 = mat2.m31
        b14 = mat2.m41
        b21 = mat2.m12
        b22 = mat2.m22
        b23 = mat2.m32
        b24 = mat2.m42
        b31 = mat2.m13
        b32 = mat2.m23
        b33 = mat2.m33
        b34 = mat2.m43
        b41 = mat2.m14
        b42 = mat2.m24
        b43 = mat2.m34
        b44 = mat2.m44

        return Matrix4x4( 
            a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41,
            a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42,
            a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43,
            a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44,

            a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41,
            a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42,
            a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43,
            a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44,

            a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41,
            a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42,
            a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43,
            a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44,

            a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41,
            a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42,
            a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43,
            a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44,
        )
    
    @staticmethod
    def three_create_look_at(eye : Vector3.Vector3, target : Vector3.Vector3,
        up : Vector3.Vector3) -> Matrix4x4: 

        z_axis = eye - target
        if z_axis.length_squared() == 0: 
            z_axis.z = 1

        z_axis = Vector3.Vector3.normalize(z_axis)
        x_axis = Vector3.Vector3.cross(up, z_axis)

        if x_axis.length_squared() == 0: 
            if abs( up.z) == 0: 
                z_axis.x += 0.0001
            else:
                z_axis.z += 0.0001

            z_axis = Vector3.Vector3.normalize(z_axis)
            x_axis = Vector3.Vector3.cross(up, z_axis)

        x_axis = Vector3.Vector3.normalize(x_axis)
        y_axis = Vector3.Vector3.cross(z_axis, x_axis)

        return Matrix4x4(
                m11 = x_axis.x,
                m12 = x_axis.y,
                m13 = x_axis.z,
                m14 = 0.0,

                m21 = y_axis.x,
                m22 = y_axis.y,
                m23 = y_axis.z,
                m24 = 0.0,

                m31 = z_axis.x,
                m32 = z_axis.y,
                m33 = z_axis.z,
                m34 = 0.0,

                m41 = 0.0,
                m42 = 0.0,
                m43 = 0.0,
                m44 = 1,
        )