# This file:
# Contains the main application class. Imports are made on a case-by-case basis, to avoid extra import time.
# run() follows these steps:
# * Part 0      -- Checks and info;
# * Part 1      -- Get the desired datasets as a pandas dataFrames
# * * * At the end of the p1_get_datasets_as_dataFrames() there will either be a self.dataFrameMap (df + label) or an error;
# * Part 1.5    -- Display dataset informations
# * Part 2      -- Get the suitable solution(s) (by user input or by APES's recommendation);
# * Part 3      -- Run selected solution(s);


from apes_dataset_handler import *
from helpers_aiders_and_conveniencers.misc_functions import *

# TODO:
# * check solution_nature against dataset_metadata.is_labeled and process dataset if necessary


class APES_Application:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger
        self.application_instance_metadata = application_instance_metadata

        info_message = "Created object of type APES_Application"
        self.Logger.info(self, info_message)
        info_message = "Starting run. This is a mock-up run"
        self.Logger.info(self, info_message)

    def run(self):
        self.application_instance_metadata.printMetadata()

        ## Part 0 -- Checks and info
        return_code, return_message = self.p0_checks()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        ## Part 1 -- Get the desired datasets as a pandas dataFrames
        return_code, return_message = self.p1_get_datasets_as_dataFrames()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        ## Part 2 -- Get the desired solution(s) runners
        return_code, return_message = self.p2_get_desired_solutions()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        ## Part 3 -- Go to town with the solution(s)
        return_code, return_message = self.p3_run_solutions()
        if return_code != 0:
            return (return_code, return_message)
        else:
            self.Logger.info(self, return_message)

        return (0, "Program exited successfully")

    def p0_checks(self):
        if (
            self.application_instance_metadata.application_mode == "compare_solutions"
            and (
                self.application_instance_metadata.solution_category == "ral"
                or self.application_instance_metadata.solution_category == "img"
            )
        ):
            info_message = "APES app set to run in application_mode = 'compare_solutions' with solution_category = 'ral' or 'img', but we only have one solution of each"
            self.Logger.info(self, info_message)

            return (1, "application_mode incompatible with solution_type")

        if (
            self.application_instance_metadata.dataset_category
            != self.application_instance_metadata.solution_category
        ):
            info_message = "Dataset category and solution category do not match"
            self.Logger.info(self, info_message)

            return (1, "dataset_category and solution_category do not match")

        return (0, "Function p0_checks exited successfully")

    def p1_get_datasets_as_dataFrames(self):
        if self.application_instance_metadata.dataset_origin == "new_dataset":
            info_message = "Dataset is not coming from our database."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner"
                self.Logger.info(self, info_message)

                from apes_discerner import Discerner

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                (
                    return_code,
                    return_message,
                    self.dataFramesList,
                    self.dataFramesUtilityList,
                ) = handle_dataset_from_path(
                    self.Logger, self.application_instance_metadata
                )
                if return_code != 0:
                    return (return_code, return_message)
                else:
                    self.Logger.info(self, return_message)
                pass

        else:
            info_message = "Dataset is coming from our database"
            self.Logger.info(self, info_message)

            # We don't have unidentified databases at the moment of writing this code, but just in case.
            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner"
                self.Logger.info(self, info_message)

                from apes_discerner import Discerner

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                (
                    return_code,
                    return_message,
                    self.dataFramesList,
                    self.dataFramesUtilityList,
                ) = handle_dataset_from_path(
                    self.Logger, self.application_instance_metadata
                )
                if return_code != 0:
                    return (return_code, return_message)
                else:
                    self.Logger.info(self, return_message)
                pass

        info_message = "p1_get_datasets_as_dataFrames returned the following:"
        for i in range(0, len(self.dataFramesList)):
            info_message += f"\nDataset {i}\n|___label = {self.dataFramesUtilityList[i]}\n|___shape = {self.dataFramesList[i].shape}"
        self.Logger.info(self, info_message)

        return (0, "Function p1_get_datasets_as_dataFrames exited successfully")

    def p15_display_dataset_informations(self):
        pass

    def p2_get_desired_solutions(self):
        solution_indexes = list(self.application_instance_metadata.solution_index)
        self.solutions_list = []
        # if self.application_instance_metadata.application_mode == "run_one_solution":
        if len(solution_indexes) == 1:
            info_message = "Only one solution to load."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.solution_category == "ral":
                info_message = "Loading solution ral_1"
                self.Logger.info(self, info_message)
                # TODO: actually implement the solution lol
                pass
            elif self.application_instance_metadata.solution_category == "img":
                info_message = "Loading solution img_1"
                self.Logger.info(self, info_message)
                # TODO: actually implement the solution lol
                pass
            else:
                match solution_indexes[0]:
                    case 1:
                        info_message = "Loading supervised solution ekg_1"
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_1 import Solution_ekg_1

                        self.solution = Solution_ekg_1(
                            self.application_instance_metadata,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution)

                        return (
                            0,
                            "Function p2_get_desired_solutions exited successfully",
                        )
                    case 2:
                        info_message = "Loading supervised solution ekg_2"
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_2 import Solution_ekg_2

                        self.solution = Solution_ekg_2(
                            self.application_instance_metadata,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution)

                        return (
                            0,
                            "Function p2_get_desired_solutions exited successfully",
                        )
                    case 3:
                        return (
                            1,
                            "Function p2_get_desired_solutions exited like a piece of shit because we don't have an ekg3 solution",
                        )

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
                        info_message = "Loading solutions ekg_1 and ekg_2."
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_1 import Solution_ekg_1

                        info_message = "Loaded Solution_ekg_1."
                        self.Logger.info(self, info_message)

                        from type_ekg.apes_solution_ekg_2 import Solution_ekg_2

                        info_message = "Loaded Solution_ekg_2."
                        self.Logger.info(self, info_message)

                        self.solution_1 = Solution_ekg_1(
                            self.application_instance_metadata,
                            self.Logger,
                        )
                        self.solution_2 = Solution_ekg_2(
                            self.application_instance_metadata,
                            self.Logger,
                        )
                        self.solutions_list.append(self.solution_1)
                        self.solutions_list.append(self.solution_2)

                        return (
                            0,
                            "Function p2_get_desired_solutions exited successfully",
                        )
                    case other:
                        return (
                            1,
                            "Function p2_get_desired_solutions exited like a piece of shit because we don't have an ekg3 solution",
                        )

        return (
            0,
            "Function p2_get_desired_solutions exited successfully, although this message should not be shown",
        )

    def p3_run_solutions(self):
        for solution in self.solutions_list:
            info_message = f"Beginning run for solution {solution}"
            self.Logger.info(self, info_message)
            solution.adapt_dataset(
                self.application_instance_metadata,
                self.dataFramesList,
                self.dataFramesUtilityList,
            )

            if self.application_instance_metadata.model_origin == "train_new_model":
                info_message = f"Directive to train new model"
                self.Logger.info(self, info_message)
                solution.create_model()
                solution.train(self.application_instance_metadata.model_train_epochs)
                solution.save_model()
                solution.test()
            else:
                ## check if sizes compatible with dataset's sizes
                info_message = f"Directive to use existing model"
                self.Logger.info(self, info_message)
                solution.load_model()
                solution.test()

        return 0, "Function p3_run_solutions exited successfully"
