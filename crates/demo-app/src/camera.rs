use std::ops::AddAssign;

use nalgebra_glm as glm;

pub struct Camera {
    position: glm::Vec3,
    front: glm::Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: glm::vec3(0.0, 0.0, 0.0),
            front: glm::vec3(0.0, 0.0, -1.0),
            yaw: -90.0,
            pitch: 0.0,
        }
    }

    pub fn get_front(&self) -> glm::Vec3 {
        self.front
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position = glm::vec3(x, y, z);
    }

    pub fn get_position(&self) -> glm::Vec3 {
        self.position
    }

    pub fn set_rotation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-89.99, 89.99);

        self.front = glm::normalize(&glm::vec3(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        ));
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        self.position.add_assign(z * self.front);
        self.position
            .add_assign(glm::normalize(&glm::cross(&self.front, &glm::vec3(0.0, 1.0, 0.0))) * x);
    }

    pub fn rotate(&mut self, yaw: f32, pitch: f32) {
        self.set_rotation(self.yaw + yaw, self.pitch + pitch);
    }
}
