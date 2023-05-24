class Apes_application:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger
        self.application_instance_metadata = application_instance_metadata

        info_message = "Created object of type Apes_application"
        self.Logger.info(self, info_message)
        info_message = "Starting run. This is a mock-up run."
        self.Logger.info(self, info_message)

    def run(self):
        self.application_instance_metadata.printMetadata()
