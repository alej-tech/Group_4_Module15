from apps.gradio.Group4_Module15_GradioApp import create_app

app = create_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
