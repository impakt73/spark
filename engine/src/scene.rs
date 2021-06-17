use std::time::Duration;

use serde::{Deserialize, Serialize};
use splines::{Interpolate, Interpolation, Key, Spline};

trait Property {
    fn get_name(&self) -> &str;
    fn get_description(&self) -> &str;
}

#[derive(Serialize, Deserialize)]
struct StaticProperty<T> {
    name: String,
    description: String,
    value: T,
}

impl<T> StaticProperty<T> {
    pub fn new(name: &str, description: &str, value: T) -> Self {
        let name = name.to_owned();
        let description = description.to_owned();
        Self {
            name,
            description,
            value,
        }
    }

    pub fn get_value(&self) -> &T {
        &self.value
    }

    pub fn set_value(&mut self, value: T) {
        self.value = value;
    }
}

impl<T> Property for StaticProperty<T> {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_description(&self) -> &str {
        &self.description
    }
}

#[derive(Serialize, Deserialize)]
struct DynamicProperty<T: Interpolate<f32>> {
    name: String,
    description: String,
    values: Spline<f32, T>,
}

impl<T: Interpolate<f32>> DynamicProperty<T> {
    pub fn new(name: &str, description: &str, value: T) -> Self {
        let name = name.to_owned();
        let description = description.to_owned();
        let values = Spline::from_vec(vec![Key::new(0.0 as f32, value, Interpolation::Linear)]);
        Self {
            name,
            description,
            values,
        }
    }

    pub fn get_value_at(&self, time: Duration) -> T {
        self.values.clamped_sample(time.as_secs_f32()).unwrap()
    }

    pub fn set_value_at(&mut self, time: Duration, value: T) {
        if let Some((idx, _)) = self
            .values
            .keys()
            .iter()
            .enumerate()
            .find(|(_, key)| key.t == time.as_secs_f32())
        {
            *self.values.get_mut(idx).unwrap().value = value;
        }
    }
}

impl<T: Interpolate<f32>> Property for DynamicProperty<T> {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_description(&self) -> &str {
        &self.description
    }
}

trait Entity {
    fn get_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PROPERTY_NAME: &str = "Test Property Name";
    const TEST_PROPERTY_DESCRIPTION: &str = "Test Property Description";

    #[test]
    fn static_property_basic() {
        let init_value = 0xdeadbeef as u32;
        let test_value = 0x12345678 as u32;

        let mut prop =
            StaticProperty::new(TEST_PROPERTY_NAME, TEST_PROPERTY_DESCRIPTION, init_value);

        assert_eq!(prop.get_name(), TEST_PROPERTY_NAME);
        assert_eq!(prop.get_description(), TEST_PROPERTY_DESCRIPTION);

        assert_eq!(*prop.get_value(), init_value);

        prop.set_value(test_value);
        assert_eq!(*prop.get_value(), test_value);
    }

    #[test]
    fn dynamic_property_basic() {
        let init_value = 0.0 as f32;
        let test_value = 1.0 as f32;

        let time = Duration::from_secs_f32(0.0);

        let mut prop =
            DynamicProperty::new(TEST_PROPERTY_NAME, TEST_PROPERTY_DESCRIPTION, init_value);

        assert_eq!(prop.get_name(), TEST_PROPERTY_NAME);
        assert_eq!(prop.get_description(), TEST_PROPERTY_DESCRIPTION);

        assert_eq!(prop.get_value_at(time), init_value);

        prop.set_value_at(time, test_value);
        assert_eq!(prop.get_value_at(time), test_value);
    }
}
