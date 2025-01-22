from locust import HttpUser, task, between
import numpy as np
from PIL import Image
import random
import io


class LocustUser(HttpUser):
    wait_time = between(1, 5)

    def generate_payload(self):
        width = random.randint(500, 2000)
        height = random.randint(500, 2000)

        random_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        image_pil = Image.fromarray(random_image)
        image_io = io.BytesIO()
        image_pil.save(image_io, format="PNG")
        image_io.seek(0)

        files = {
            "file": ("random_image.png", image_io.getvalue(), "image/png")
        }

        return files

    @task
    def sanity_check(self):
        self.client.get("/")

    @task
    def prediction(self):
        files = self.generate_payload()
        self.client.post("/predictions", files=files)

    @task
    def probabilities(self):
        files = self.generate_payload()
        self.client.post("/probabilities", files=files)

    @task
    def annotations(self):
        files = self.generate_payload()
        self.client.post("/annotations", files=files)