use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::thread;
use std::{fmt, fs::File, io::BufReader};
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::JoinHandle,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
enum AudioRequest {
    Exit,
    LoadTrack(String),
    Play,
    Pause,
}

#[derive(Debug)]
enum AudioResponse {
    Result(bool),
}

#[derive(Debug, Clone)]
enum AudioDeviceError {
    InvalidResponse,
}

impl std::error::Error for AudioDeviceError {}

impl fmt::Display for AudioDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#?}", &self)
    }
}

#[derive(Debug)]
enum AudioPacket {
    Request(AudioRequest),
    Response(AudioResponse),
}

fn run_server(send: SyncSender<AudioPacket>, recv: Receiver<AudioPacket>) -> Result<()> {
    let (stream, handle) = OutputStream::try_default()?;
    let sink = Sink::try_new(&handle)?;
    sink.pause();

    let mut should_exit = false;

    while !should_exit {
        let packet = recv.recv()?;
        println!("Got packet: {:#?}", packet);
        let mut result = true;
        if let AudioPacket::Request(request) = packet {
            match request {
                AudioRequest::Exit => {
                    should_exit = true;
                }
                AudioRequest::Play => {
                    sink.play();
                }
                AudioRequest::Pause => {
                    sink.pause();
                }
                AudioRequest::LoadTrack(track_path) => {
                    let file = File::open(&track_path)?;
                    let source = rodio::Decoder::new(BufReader::new(file))?;
                    sink.append(source);
                    result = true;
                }
                _ => {
                    result = false;
                }
            }
        }
        send.send(AudioPacket::Response(AudioResponse::Result(result)))?;
    }

    Ok(())
}

pub struct AudioDevice {
    thread: Option<JoinHandle<()>>,
    send: SyncSender<AudioPacket>,
    recv: Receiver<AudioPacket>,
}

impl AudioDevice {
    pub fn new() -> Result<Self> {
        let (server_send, recv) = mpsc::sync_channel::<AudioPacket>(0);
        let (send, server_recv) = mpsc::sync_channel::<AudioPacket>(0);
        let thread = thread::spawn(move || {
            run_server(server_send, server_recv).expect("Failed to run audio server thread");
        });

        Ok(Self {
            thread: Some(thread),
            send,
            recv,
        })
    }

    pub fn load_track(&mut self, track_path: &str) {
        self.transact(AudioRequest::LoadTrack(String::from(track_path)))
            .expect("Failed to load audio device track");
    }

    pub fn play(&mut self) {
        self.transact(AudioRequest::Play)
            .expect("Failed to play audio device");
    }

    pub fn pause(&mut self) {
        self.transact(AudioRequest::Pause)
            .expect("Failed to pause audio device");
    }

    fn transact(&self, request: AudioRequest) -> Result<AudioResponse> {
        self.send.send(AudioPacket::Request(request))?;
        let packet = self.recv.recv()?;
        match packet {
            AudioPacket::Response(response) => Ok(response),
            _ => Err(Box::new(AudioDeviceError::InvalidResponse)),
        }
    }
}

impl Drop for AudioDevice {
    fn drop(&mut self) {
        self.transact(AudioRequest::Exit)
            .expect("Failed to request audio device exit");
        self.thread
            .take()
            .unwrap()
            .join()
            .expect("Failed to join audio device thread");
    }
}
