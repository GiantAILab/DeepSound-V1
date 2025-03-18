# coding=utf-8

class PostProcess:
    def init(self, v2a_processor):
        self.v2a_processor = v2a_processor

    def run(self, video_path, prompt, neg_prompt, raw_audio=None, regen=True, mode: str="neg"):
        assert mode in ['rm', 'rep', 'neg'], f"Error: Invalid 'mode' value '{mode}'. Valid values are 'rm', 'rep', 'neg'."
        if mode == "rm":
            return None
        elif mode == 'rep':
            if raw_audio is not None:
                return raw_audio
            elif raw_audio is None and regen:
                print("Regenerate audio...")
                audio = self.v2a_processor.generate(video_path, prompt)
                return audio
            else:
                return None
        elif mode == 'neg':
            print(f"Regenerate audio using {neg_prompt}")
            audio = self.v2a_processor.generate(video_path, prompt, neg_prompt)
            return audio
        
        return audio

    