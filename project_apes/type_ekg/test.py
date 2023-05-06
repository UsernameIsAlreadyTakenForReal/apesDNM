class test:
    def __init__(self, shared_definitions):
        self.project_solution_model_filename = (
            shared_definitions.project_solution_ekg_1_model_filename
        )
        self.project_solution_training_script = (
            shared_definitions.project_solution_ekg_1_training_script
        )

    def get_project_solution_model_filename(self):
        print(self.project_solution_model_filename)

    def get_project_solution_training_script(self):
        print(self.project_solution_training_script)
