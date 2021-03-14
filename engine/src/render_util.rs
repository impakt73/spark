use std::io;
use std::slice;

/// Utility structure that simplifies the process of writing  that's constant over a single frame into GPU memory
pub struct ConstantDataWriter {
    p_buffer: *mut u8,
    buffer_size: usize,
    bytes_written: usize,
}

impl ConstantDataWriter {
    pub fn new() -> Self {
        ConstantDataWriter {
            p_buffer: std::ptr::null_mut(),
            buffer_size: 0,
            bytes_written: 0,
        }
    }

    pub fn begin_frame(&mut self, p_buffer: *mut u8, buffer_size: usize) {
        self.p_buffer = p_buffer;
        self.buffer_size = buffer_size;
        self.bytes_written = 0;
    }

    pub fn end_frame(&mut self) -> usize {
        self.bytes_written
    }

    pub fn dword_offset(&self) -> u32 {
        (self.bytes_written / 4) as u32
    }
}

impl Default for ConstantDataWriter {
    fn default() -> Self {
        ConstantDataWriter::new()
    }
}

impl io::Write for ConstantDataWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_remaining = self.buffer_size - self.bytes_written;
        let bytes_written = if buf.len() <= bytes_remaining {
            buf.len()
        } else {
            bytes_remaining
        };

        let buffer = unsafe {
            slice::from_raw_parts_mut(self.p_buffer.add(self.bytes_written), bytes_remaining)
        };
        buffer[..bytes_written].clone_from_slice(&buf[..bytes_written]);

        self.bytes_written += bytes_written;

        Ok(bytes_written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
