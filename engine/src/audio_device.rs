use bass_sys as sys;

use std::{
    error::Error,
    ffi::{self, c_void},
    fmt,
    path::Path,
    ptr,
    sync::Arc,
    time::Duration,
};

#[derive(Debug)]
pub struct AudioError {
    error_code: i32,
}

impl AudioError {
    fn from_last_error() -> Option<Self> {
        let error_code = unsafe { sys::BASS_ErrorGetCode() };
        if error_code != 0 {
            Some(Self { error_code })
        } else {
            None
        }
    }
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BASS Error Code: {}", self.error_code)
    }
}

impl Error for AudioError {}

type AudioResult<T> = Result<T, AudioError>;

macro_rules! sys_call_handle {
    ($call:expr) => {
        if $call != 0 {
            Ok($call)
        } else {
            Err(AudioError::from_last_error().unwrap())
        }
    };
}

macro_rules! sys_call_uint {
    ($call:expr) => {
        if $call != 0xFFFF_FFFF {
            Ok($call)
        } else {
            Err(AudioError::from_last_error().unwrap())
        }
    };
}

macro_rules! sys_call_float {
    ($call:expr) => {
        if $call >= 0.0 {
            Ok($call)
        } else {
            Err(AudioError::from_last_error().unwrap())
        }
    };
}

macro_rules! sys_call_bool {
    ($call:expr) => {
        if $call == 0 {
            Err(AudioError::from_last_error().unwrap())
        } else {
            Ok(())
        }
    };
}

#[derive(Debug, PartialEq, Eq)]
pub enum AudioTrackState {
    Stopped,
    Playing,
    Stalled,
    Paused,
    PausedDevice,
}

impl From<u32> for AudioTrackState {
    fn from(value: u32) -> Self {
        match value {
            sys::BASS_ACTIVE_STOPPED => Self::Stopped,
            sys::BASS_ACTIVE_PLAYING => Self::Playing,
            sys::BASS_ACTIVE_STALLED => Self::Stalled,
            sys::BASS_ACTIVE_PAUSED => Self::Paused,
            sys::BASS_ACTIVE_PAUSED_DEVICE => Self::PausedDevice,
            _ => panic!("Invalid audio track state value!"),
        }
    }
}

/// Returns the duration of the provided stream
///
/// Note: This function will only work if the provided stream was created with the BASS_STREAM_PRESCAN flag.
fn calc_stream_duration(stream: sys::HSTREAM) -> AudioResult<Duration> {
    let length_byte =
        unsafe { sys_call_uint!(sys::BASS_ChannelGetLength(stream, sys::BASS_POS_BYTE)) }?;
    let length_secs =
        unsafe { sys_call_float!(sys::BASS_ChannelBytes2Seconds(stream, length_byte)) }?;
    Ok(Duration::from_secs_f64(length_secs))
}

pub struct AudioTrack {
    stream: sys::HSTREAM,
    duration: Duration,

    // Used to keep the associated audio device alive
    #[allow(dead_code)]
    device: Arc<AudioDevice>,
}

impl AudioTrack {
    pub fn from_path(
        device: Arc<AudioDevice>,
        path: &(impl AsRef<Path> + ?Sized),
    ) -> AudioResult<Self> {
        let path = ffi::CString::new(path.as_ref().to_str().unwrap()).unwrap();
        let stream = unsafe {
            sys_call_handle!(sys::BASS_StreamCreateFile(
                0,
                path.as_ptr() as *const c_void,
                0,
                0,
                sys::BASS_STREAM_PRESCAN
            ))
        }?;
        let duration = calc_stream_duration(stream)?;
        Ok(Self {
            stream,
            duration,
            device,
        })
    }

    pub fn play(&mut self) -> AudioResult<()> {
        unsafe { sys_call_bool!(sys::BASS_ChannelPlay(self.stream, 0)) }
    }

    pub fn pause(&mut self) -> AudioResult<()> {
        unsafe { sys_call_bool!(sys::BASS_ChannelPause(self.stream)) }
    }

    pub fn stop(&mut self) -> AudioResult<()> {
        unsafe { sys_call_bool!(sys::BASS_ChannelStop(self.stream)) }
    }

    pub fn get_state(&self) -> AudioTrackState {
        unsafe { sys::BASS_ChannelIsActive(self.stream) }.into()
    }

    pub fn is_playing(&self) -> bool {
        self.get_state() == AudioTrackState::Playing
    }

    pub fn toggle_pause(&mut self) -> AudioResult<()> {
        match self.get_state() {
            AudioTrackState::Paused | AudioTrackState::PausedDevice => self.play(),
            AudioTrackState::Playing => self.pause(),
            AudioTrackState::Stopped => {
                let pos = self.get_position()?;
                let last_pos = self.duration - Duration::from_millis(1);
                if pos >= last_pos {
                    // We're currently stopped at the end of the track
                    // Don't do anything here or the track will loop and that's desired ideal behavior.
                    Ok(())
                } else {
                    // The track isn't currently playing, but we're at a valid track position
                    // Just play as normal in this case
                    self.play()
                }
            }
            _ => panic!("Unexpected BASS track state"),
        }
    }

    pub fn get_position(&self) -> AudioResult<Duration> {
        let pos_byte = unsafe {
            sys_call_uint!(sys::BASS_ChannelGetPosition(
                self.stream,
                sys::BASS_POS_BYTE
            ))
        }?;
        let pos_secs =
            unsafe { sys_call_float!(sys::BASS_ChannelBytes2Seconds(self.stream, pos_byte)) }?;
        Ok(Duration::from_secs_f64(pos_secs))
    }

    pub fn set_position(&mut self, pos: &Duration) -> AudioResult<()> {
        let pos_byte = unsafe {
            sys_call_uint!(sys::BASS_ChannelSeconds2Bytes(
                self.stream,
                pos.as_secs_f64()
            ))
        }?;
        unsafe {
            sys_call_bool!(sys::BASS_ChannelSetPosition(
                self.stream,
                pos_byte,
                sys::BASS_POS_BYTE
            ))
        }?;
        Ok(())
    }

    pub fn add_position_offset(&mut self, pos: &Duration) -> AudioResult<()> {
        let cur_pos = self.get_position()?;
        let pos = cur_pos + *pos;

        // Clamp to a tiny bit before the end of the track
        let last_pos = self.duration - Duration::from_millis(1);
        let pos = std::cmp::min(pos, last_pos);

        self.set_position(&pos)
    }

    pub fn subtract_position_offset(&mut self, pos: &Duration) -> AudioResult<()> {
        let cur_pos = self.get_position()?;
        // Clamp to 0
        let pos = cur_pos.checked_sub(*pos).unwrap_or_else(Duration::default);
        self.set_position(&pos)
    }

    pub fn get_length(&self) -> Duration {
        self.duration
    }
}

impl Drop for AudioTrack {
    fn drop(&mut self) {
        unsafe {
            sys_call_bool!(sys::BASS_StreamFree(self.stream)).expect("Failed to free BASS stream");
        }
    }
}

pub struct AudioDevice {}

pub struct AudioDeviceConfig {
    frequency: u32,
}

impl Default for AudioDeviceConfig {
    fn default() -> Self {
        Self { frequency: 48000 }
    }
}

impl AudioDevice {
    pub fn new() -> AudioResult<Self> {
        Self::new_with_config(AudioDeviceConfig::default())
    }

    pub fn new_with_config(config: AudioDeviceConfig) -> AudioResult<Self> {
        unsafe {
            sys_call_bool!(sys::BASS_Init(
                -1,
                config.frequency,
                0,
                ptr::null_mut(),
                ptr::null_mut()
            ))?;
            Ok(Self {})
        }
    }
}

impl Drop for AudioDevice {
    fn drop(&mut self) {
        unsafe {
            sys_call_bool!(sys::BASS_Free()).expect("Failed to free BASS audio device");
        }
    }
}
