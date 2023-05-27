# This file:
# Contains the main application class. Imports are made on a case-by-case basis, to avoid extra import time.
# run() follows these steps:
# * Part 0 -- Checks and info;
# * Part 1 -- Get the desired dataset as a pandas dataFrame
# * * * At the end of the p1_getDataset_asDataFrame() there will either be a self.dataFrameList or an error.
# * Part 2 -- Get the suitable solution(s) (by user input or by apes's recommendation);
# * Part 3 -- Run selected solution(s);


from apes_dataset_handler import *

# TODO:
# * check solution_nature against dataset_metadata.is_labeled and process dataset if necessary


class APES_Application:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger
        self.application_instance_metadata = application_instance_metadata

        info_message = "Created object of type APES_Application."
        self.Logger.info(self, info_message)
        info_message = "Starting run. This is a mock-up run."
        self.Logger.info(self, info_message)

    def run(self):
        self.application_instance_metadata.printMetadata()

        ## Part 0 -- Checks and info
        return_code, return_message = self.p0_checks()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        ## Part 1 -- Get the desired dataset as a pandas dataFrame
        return_code, return_message = self.p1_getDataset_asDataFrame()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        ## Part 2 -- Get the desired solution(s) runners
        solution_indexes = list(self.application_instance_metadata.solution_index)
        self.solutions_list = []
        # if self.application_instance_metadata.application_mode == "run_one_solution":
        if len(solution_indexes) == 1:
            info_message = "Only one solution to load."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.solution_category == "ral":
                info_message = "Loading solution ral_1"
                self.Logger.info(self, info_message)
                pass
            elif self.application_instance_metadata.solution_category == "img":
                info_message = "Loading solution img_1"
                self.Logger.info(self, info_message)
                pass
            else:
                match solution_indexes[0]:
                    case 1:
                        info_message = "Loading supervised solution ekg_1"
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_1_train import Solution_ekg_1

                        self.solution = Solution_ekg_1(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution)
                        pass
                    case 2:
                        info_message = "Loading solution ekg_2"
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_2_train import Solution_ekg_2

                        self.solution = Solution_ekg_2(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution)
                        pass

        else:
            info_message = "Multiple solutions to load."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.solution_category == "ral":
                info_message = "Beep beep, I'm a sheep. There's only 1 ral solution. Why is there a directive to load more?"
                self.Logger.info(self, info_message)
            elif self.application_instance_metadata.solution_category == "img":
                info_message = "Beep beep, I'm a sheep. There's only 1 img solution. Why is there a directive to load more?"
                self.Logger.info(self, info_message)
            else:
                match solution_indexes:
                    case [1, 2]:
                        info_message = (
                            "Multiple solutions to load. Loading ekg_1 and ekg_2."
                        )
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_1_train import Solution_ekg_1
                        from type_ekg.apes_solution_ekg_2_train import Solution_ekg_2

                        self.solution_1 = Solution_ekg_1(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        self.solution_2 = Solution_ekg_2(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution_1)
                        self.solutions_list.append(self.solution_2)
                        pass

        ## Part 3 -- Go to town with the solution(s)
        for solution in self.solutions_list:
            solution.adapt_dataset(self.application_instance_metadata, self.dataFrame)

            # if self.application_instance_metadata.model_origin == "train_new_model":
            #     solution.create_model()
            #     solution.train()
            #     solution.save_model()
            #     solution.test()
            #     pass
            # else:
            #     ## check if sizes compatible with dataset's sizes
            #     solution.load_model()
            #     solution.test()
            #     pass

        return (0, "Program exited successfully")

    def p0_checks(self):
        ## Fail if compare_solutions + solution_category==ral or solution_category==img (subject to change)
        if (
            self.application_instance_metadata.application_mode == "compare_solutions"
            and (
                self.application_instance_metadata.solution_category == "ral"
                or self.application_instance_metadata.solution_category == "img"
            )
        ):
            info_message = "APES app set to run in application_mode = 'compare_solutions' with solution_category = 'ral' or 'img', but we only have one solution of each."
            self.Logger.info(self, info_message)

            return (1, "application_mode incompatible with solution_type.")

        if (
            self.application_instance_metadata.dataset_category
            != self.application_instance_metadata.solution_category
        ):
            info_message = "Dataset category and solution category do not match."
            self.Logger.info(self, info_message)

            return (1, "dataset_category and solution_category do not match.")

        return (0, "Function p0_checks exited successfully")

    def p1_getDataset_asDataFrame(self):
        if self.application_instance_metadata.dataset_origin == "new_dataset":
            info_message = "Dataset is not coming from our database."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner."
                self.Logger.info(self, info_message)

                from apes_discerner import Discerner

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                self.dataFrame = process_file_type(self.application_instance_metadata)
                pass

        else:
            info_message = "Dataset is coming from our database."
            self.Logger.info(self, info_message)

            # We don't have unidentified databases at the moment of writing this code, but just in case.
            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner."
                self.Logger.info(self, info_message)

                from apes_discerner import Discerner

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                self.dataFrame = process_file_type(
                    self.Logger, self.application_instance_metadata
                )
                print(self.application_instance_metadata.dataset_metadata.dataset_path)
                pass
        return (0, "Function p1_getDataset_asDataFrame exited successfully")
