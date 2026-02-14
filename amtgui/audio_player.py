from PySide6.QtCore import Qt, QObject, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider
    )

class AudioPlayer(QObject):
    def __init__(self):
        super().__init__()

        self.play_button = QPushButton("Play")
        self.play_button.pressed.connect(self.play_button_clicked)

        self.stop_button = QPushButton("Stop")
        self.stop_button.pressed.connect(self.stop_button_clicked)

        self.reset_button = QPushButton("Reset")
        self.reset_button.pressed.connect(self.reset_button_clicked)

        self.pause_button = QPushButton("Pause")
        self.pause_button.pressed.connect(self.pause_button_clicked)

        self.forward_button = QPushButton("Forward")
        self.forward_button.pressed.connect(self.forward_button_clicked)

        self.backward_button = QPushButton("Backward")
        self.backward_button.pressed.connect(self.backward_button_clicked)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.volume_slider_changed)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderMoved.connect(self.set_position)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)

        self.top_layout = QHBoxLayout()
        self.top_layout.addWidget(self.play_button)
        self.top_layout.addWidget(self.stop_button)
        self.top_layout.addWidget(self.reset_button)
        self.top_layout.addWidget(self.pause_button)

        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.forward_button)
        self.bottom_layout.addWidget(self.backward_button)
        self.bottom_layout.addWidget(self.volume_slider)

        self.audio_buttons_layout = QVBoxLayout()
        self.audio_buttons_layout.addLayout(self.top_layout)
        self.audio_buttons_layout.addWidget(self.progress_slider)
        self.audio_buttons_layout.addLayout(self.bottom_layout) 

        self.audio_control_buttons = QWidget()
        self.audio_control_buttons.setLayout(self.audio_buttons_layout)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.audio_output.setVolume(1.0)

        self._is_slider_active = False

    def load_audio(self, file_path):
        self.player.setSource(QUrl.fromLocalFile(file_path))

    def play_button_clicked(self):
        self.player.play()
    
    def stop_button_clicked(self):
        self.player.stop()
    
    def reset_button_clicked(self):
        self.player.stop()
        self.player.setPosition(0)

    def pause_button_clicked(self):
        self.player.pause()

    def forward_button_clicked(self):
        self.player.setPosition(self.player.position() + 5000)

    def backward_button_clicked(self):
        self.player.setPosition(self.player.position() - 5000)

    def volume_slider_changed(self, value):
        self.audio_output.setVolume(value / 100.0)

    def position_changed(self, position):
        if not self._is_slider_active:
            self.progress_slider.setValue(position)

    def duration_changed(self, duration):
        self.progress_slider.setRange(0, duration)

    def set_position(self, position):
        self.player.setPosition(position)

    def slider_pressed(self):
        self._is_slider_active = True
        self.player.pause()

    def slider_released(self):
        self._is_slider_active = False
        self.set_position(self.progress_slider.value())
        self.player.play()
