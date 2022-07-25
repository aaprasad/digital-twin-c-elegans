from moviepy.editor import VideoFileClip


def speed_up(path, speed):
    """ speed up a video """
    output_path = path[0:-4] + '.new' + '.mp4'
    audio = VideoFileClip(path)
    new_audio = audio.fl_time(lambda t: speed * t, apply_to=['mask', 'audio'])
    new_audio = new_audio.set_duration(audio.duration / speed)
    new_audio.write_videofile(output_path)


if __name__ == '__main__':
    speed_up(path='openaigym.video.0.31003.video000000.mp4', speed=5)
