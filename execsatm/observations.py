from collections.abc import Iterable
import math
from typing import Dict, List, Set, Tuple, Union
import uuid

from functools import reduce
import numpy as np

from execsatm.tasks import GenericObservationTask
from execsatm.objectives import MissionObjective
from execsatm.utils import EmptyInterval, Interval

class ObservationOpportunity:
    # -------------------------------------------
    #               CONSTRUCTORS
    # -------------------------------------------
    def __init__(self,
                 tasks : Set[GenericObservationTask],
                 instrument_name : str, 
                 task_accessibility : Dict[str, Interval],
                 task_slew_angles : Dict[str, Interval],
                 task_min_duration : Dict[str, float],
                 accessibility : Interval,
                 slew_angles : Interval, 
                 min_duration : float,
                 max_duration : float,
                 id : str = None,
                ):
        """
        Represents an observation opportunity that can be scheduled by an agent in order to fulfill one or more generic observation tasks.
        **Arguments:**
        """
        
        # validate input types
        assert tasks and isinstance(tasks, Iterable) and all(isinstance(task, GenericObservationTask) for task in tasks), "tasks must be a non-empty set of `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(task_accessibility, dict) and all(isinstance(task, str) and isinstance(interval, Interval) for task, interval in task_accessibility.items()), "task_accessibilities must be a dictionary mapping `GenericObservationTask` to `Interval`."
        assert isinstance(task_slew_angles, dict) and all(isinstance(task, str) and isinstance(interval, Interval) for task, interval in task_slew_angles.items()), "task_slew_angles must be a dictionary mapping `GenericObservationTask` to `Interval`."
        assert isinstance(task_min_duration, dict) and all(isinstance(task, str) and isinstance(duration, (float, int)) for task, duration in task_min_duration.items()), "min_duration_reqs must be a dictionary mapping `GenericObservationTask` to a number."     
        assert isinstance(accessibility, Interval), "Accessibility must be an `Interval` instance."
        assert isinstance(slew_angles, Interval), "Slew angles must be an `Interval` instance."
        assert isinstance(min_duration, (float, int)), "Minimum duration must be a number."
        assert isinstance(max_duration, (float, int)), "Maximum duration must be a number."
        assert isinstance(id, str) or id is None, "ID must be a string or `None`."
        
        # compute joint availability interval based on parent tasks' availability intervals
        availability = self.__merge_task_availability(tasks)
        
        # validate input values
        assert not availability.is_empty(), "The parent tasks of the observation opportunity must have overlapping availability intervals."
        assert len(task_accessibility) == len(tasks) == len(task_slew_angles) == len(task_min_duration), "task_accessibilities, task_slew_angles, and min_duration_reqs must have the same number of entries as tasks."
        assert all(task.id in task_accessibility for task in tasks), "Each task must have an accessibility interval specified in task_accessibilities."
        assert all(task.id in task_slew_angles for task in tasks), "Each task must have a slew angle interval specified in task_slew_angles."
        assert all(task.id in task_min_duration for task in tasks), "Each task must have a minimum duration requirement specified in min_duration_reqs."
        assert all(accessibility.overlaps(t_acc) for t_acc in task_accessibility.values()), "The observation opportunity's accessibility interval must be a subset of each task's accessibility interval."
        assert all(slew_angles.is_subset(t_slew_angles) for t_slew_angles in task_slew_angles.values()), \
            "The observation opportunity's slew angle interval must be a subset of each task's slew angle interval."
        assert all(min_duration >= t_min_duration for t_min_duration in task_min_duration.values()), "The observation opportunity's minimum duration requirement must be greater than or equal to each task's minimum duration requirement."

        assert min_duration >= 0.0, "Minimum duration must be non-negative."
        assert max_duration > 0.0, "Maximum duration must be positive."
        assert min_duration <= max_duration, "Minimum duration must not exceed maximum duration."
        assert all(availability.overlaps(task_accessibility[task.id]) for task in tasks), "Each task's accessibility interval must be a subset of the observation opportunity's availability interval."
        assert not accessibility.is_empty(), "Accessibility interval must not be empty."
        assert accessibility.is_subset(availability), "Accessibility interval must be a subset of the observation opportunity's availability interval."
        assert not slew_angles.is_empty(), "Slew angle interval must not be empty."
        
        # set parameters
        ## tasks and availability
        self.tasks = tasks
        self.id_to_task = {task.id : task for task in tasks}
        self.availability = availability
        
        ## instrument information
        self.instrument_name : str = instrument_name
        
        ## task-specific requirements
        self.task_accessibility : Dict[str, Interval] = task_accessibility
        self.task_slew_angles : Dict[str, Interval] = task_slew_angles
        self.task_min_duration : Dict[str, float] = task_min_duration
        
        ## compiled requirements for the observation opportunity
        self.accessibility : Interval = accessibility
        self.slew_angles : Interval = slew_angles
        self.min_duration : float = min_duration
        self.max_duration : float = max_duration

        ## define ID
        self.id : str = id if id is not None \
            else ObservationOpportunity.__generate_id(self.tasks, instrument_name, self.accessibility)
        
    def copy(self) -> 'ObservationOpportunity':
        """ Create a deep copy of the task. """
        return ObservationOpportunity(
            tasks = set(self.tasks),
            instrument_name = self.instrument_name,
            task_accessibility = dict(self.task_accessibility),
            task_slew_angles = dict(self.task_slew_angles),
            task_min_duration = dict(self.task_min_duration),
            accessibility = self.accessibility.copy(),
            slew_angles = self.slew_angles.copy(),
            min_duration = self.min_duration,
            max_duration = self.max_duration,
            id = self.id
        )

    # -------------------------------------------
    #             COMPARISON METHODS
    # -------------------------------------------

    def __eq__(self, other : 'ObservationOpportunity') -> bool:
        assert isinstance(other, ObservationOpportunity), \
            f"Can only compare with another `ObservationOpportunity`. is of type {type(other)}."
        
        my_dict = self.to_dict()
        other_dict = other.to_dict()

        for key in my_dict:
            if key not in other_dict:
                return False
            if my_dict[key] != other_dict[key]:
                if isinstance(my_dict[key], list) and isinstance(other_dict[key], list):
                    if len(my_dict[key]) != len(other_dict[key]):
                        return False
                    for item1, item2 in zip(my_dict[key], other_dict[key]):
                        if item1 != item2:
                            for item_key in item1:
                                if item_key not in item2 or item1[item_key] != item2[item_key]:
                                    return False
            
        
        return self.to_dict() == other.to_dict()

    def is_mutually_exclusive(self, other : 'ObservationOpportunity') -> bool:
        """ Check if two tasks are mutually exclusive. """
        # validate inputs
        assert isinstance(other, ObservationOpportunity), \
            "Argument must be an instance of `ObservationOpportunity`."

        # are mutually exclusive if they...
        return (                                    
            # are not the same observation opportunity,
            not other == self                                  
            # have common parent tasks,
            and len(self.tasks.intersection(other.tasks)) > 0
            # and have overlapping accessibility 
            and self.accessibility.overlaps(other.accessibility)
        )

    # -------------------------------------------
    #               MERGING METHODS
    # -------------------------------------------
    # @staticmethod
    # def __merge_task_time_reqs(
    #                             task_accessibilities : Dict[str, Interval],
    #                             min_duration_reqs : Dict[str, float],
    #                             must_overlap : bool = False
    #                         ) -> Tuple[Interval, float]:
    #     """
    #     Calculates the joint availability and duration requirements for a set of parent generic observation tasks if merged into a single observation opportunity.

    #     **Arguments:**
    #     - `task_accessibilities`: A dictionary mapping each parent generic observation task to the time interval during which the observation opportunity is accessible for that task.
    #     - `min_duration_reqs`: A dictionary mapping each parent generic observation task to the minimum duration required for the observation in seconds [s] for that task.
    #     - `must_overlap`: If True, the observation opportunities must overlap in accessibility to be merged.
    #     """
    #     # Calculate accessibility overlap
    #     accessibility_overlap : Interval \
    #         = reduce(lambda a, b: a.intersection(b), task_accessibilities.values())

    #     # check if accessibility intervals would need to be extended
    #     needs_extension : bool = accessibility_overlap.is_empty()

    #     # check if accessibility time windows are not allowed to be extended and is needed
    #     if needs_extension and must_overlap: 
    #         # we cannot merge the tasks; return invalid time requirements
    #         return accessibility_overlap, np.NaN

    #     # initialize merged accessibility and minimum duration requirement
    #     merged_accessibility : Interval = None
    #     min_duration_req : float = None 

    #     # Iterate through task accessibilities and duration requirements to find merged time requirements
    #     for task_id, accessibility in task_accessibilities.items():
    #         if merged_accessibility is None:
    #             merged_accessibility = accessibility.copy()
    #             min_duration_req = min_duration_reqs[task_id]
    #             continue

    #         # Determine which observation starts first
    #         preceeding_acc, proceeding_acc = sorted([accessibility, merged_accessibility])
    #         preceeding_req, proceeding_req = (min_duration_reqs[task_id], min_duration_req) if preceeding_acc == accessibility else (min_duration_req, min_duration_reqs[task_id])

    #         # Check if accesibility has any overlap
    #         if not needs_extension:  # There is an overlap between the tasks' availability
                
    #             # Calculate new observation duration requirement
    #             min_duration_req = max(preceeding_req, proceeding_req)             

    #             # Check if either accessibility is fully contained within the other
    #             if (proceeding_acc.is_subset(preceeding_acc)
    #                 and preceeding_acc.is_subset(proceeding_acc)):
    #                 merged_accessibility = accessibility_overlap.copy()

    #             elif proceeding_acc.is_subset(preceeding_acc):
    #                 # Keep preceeding_obs accessibility 
    #                 merged_accessibility = preceeding_acc.copy()                

    #                 # min_duration_req = preceeding_obs.min_duration

    #             elif preceeding_acc.is_subset(proceeding_acc):
    #                 # Keep proceeding_obs accessibility and duration requirements
    #                 merged_accessibility = proceeding_acc.copy()
    #                 # min_duration_req = proceeding_obs.min_duration                

    #             else:
    #                 # Use tasks' accessibility and duration requirements to find new accessibility bounds
    #                 accessibility_start = min(preceeding_acc.right - preceeding_req, 
    #                                         proceeding_acc.left)
    #                 accessibility_end   = max(preceeding_acc.right,
    #                                         proceeding_acc.left + proceeding_req)

    #                 # Merge accessibility
    #                 merged_accessibility = Interval(accessibility_start, accessibility_end) if accessibility_start <= accessibility_end else EmptyInterval()
            
    #         else: # There is no overlap between the tasks' availability; we can only merge by extending the accessibility window
    #             # Find new accessibility bounds
    #             accessibility_start = preceeding_acc.right - preceeding_req
    #             accessibility_end = proceeding_acc.left + proceeding_req

    #             # Extend accessibility
    #             merged_accessibility : Interval = Interval(accessibility_start, accessibility_end) if accessibility_end > accessibility_start else EmptyInterval()

    #             # Calculate new observation duration requirement
    #             min_duration_req = merged_accessibility.span()

    #         assert min_duration_req >= max(preceeding_req, proceeding_req),\
    #                 "Calculated minimum duration requirement does not satisfy parent observation opportunities' minimum duration requirements."

    #         if merged_accessibility.left < 0.0 or merged_accessibility.right < 0.0:
    #             raise ValueError(f"Calculated merged accessibility has negative bounds: {merged_accessibility}. This should not happen.")

    #     # Return Time Requirements
    #     return merged_accessibility, min_duration_req    

    def _calc_merged_time_reqs(self, other : 'ObservationOpportunity', must_overlap : bool = False) -> Tuple[Interval,float]:
        """
        Calculates the joint availability and duration requirements for two task observation opportunities if merged. 

        **Arguments:**
        - `other`: The other task observation opportunity to merge with.
        - `must_overlap`: If True, the observation opportunities must overlap in accessibility to be merged.
        """
        
        # Calculate accessibility overlap
        accessibility_overlap : Interval = self.accessibility.intersection(other.accessibility)

        # check if accessibility intervals would need to be extended
        needs_extension : bool = (accessibility_overlap.is_empty() 
                                  or accessibility_overlap.span() <= 0.0
                                  )
        
        # check if accessibility time windows are not allowed to be extended and is needed
        if needs_extension and must_overlap: 
            # we cannot merge the tasks; return invalid time requirements
            return accessibility_overlap, np.NaN, np.NaN

        # Determine which observation starts first
        preceeding_obs, proceeding_obs = sorted([self, other], key=lambda t: (t.accessibility.left, t.accessibility.span()))

        # Check if accesibility has any overlap
        if needs_extension: # There is no overlap between the tasks' availability
            # Find new accessibility bounds
            accessibility_start = preceeding_obs.accessibility.right - preceeding_obs.min_duration
            accessibility_end = proceeding_obs.accessibility.left + proceeding_obs.min_duration

            # Extend accessibility
            merged_accessibility : Interval = Interval(accessibility_start, accessibility_end) if accessibility_end > accessibility_start else EmptyInterval()

            # Calculate new observation duration requirement
            min_duration_req = merged_accessibility.span()
        
        else: # There is non-zero overlap between the tasks' availability            
            # Calculate new observation duration requirement
            min_duration_req = max(preceeding_obs.min_duration, proceeding_obs.min_duration)      

            # get observation accessibility intervals
            preceeding_accessibility = preceeding_obs.accessibility
            proceeding_accessibility = proceeding_obs.accessibility                  

            # Check if either accessibility is fully contained within the other
            if (proceeding_accessibility.is_subset(preceeding_accessibility)
                and preceeding_accessibility.is_subset(proceeding_accessibility)):
                # both accessibilities are the same; keep accessibility overlap
                merged_accessibility = accessibility_overlap.copy()

            elif (proceeding_accessibility.is_subset(preceeding_accessibility)
                  and proceeding_accessibility.left + proceeding_obs.min_duration \
                    <= preceeding_accessibility.left + preceeding_obs.min_duration):
                # proceeding observation accessibility is fully contained within the preceeding observation's accessibility
                # and preceeding observation already meets the proceeding observation's minimum duration requirement; 
                
                # keep preceeding_obs accessibility and duration requirements
                merged_accessibility = preceeding_obs.accessibility.copy()                

            elif (preceeding_accessibility.is_subset(proceeding_accessibility)
                  and preceeding_accessibility.left + preceeding_obs.min_duration \
                    <= proceeding_accessibility.left + proceeding_obs.min_duration):
                # preceeding observation accessibility is fully contained within the proceeding observation's accessibility
                # and proceeding observation already meets the preceeding observation's minimum duration requirement;

                # keep preceeding_obs accessibility and duration requirements
                merged_accessibility = proceeding_obs.accessibility.copy()    

            else:
                # Use tasks' accessibility and duration requirements to find new accessibility bounds
                accessibility_start = min(
                                        # Latest time to start preceeding observation and satisfy its minimum duration requirement
                                        preceeding_obs.accessibility.right - preceeding_obs.min_duration, 
                                        # Earliest time to start proceeding observation
                                        proceeding_obs.accessibility.left
                                    )
                accessibility_end   = max(
                                        # Latest time to end preceeding observation
                                        preceeding_obs.accessibility.right,
                                        # Earliest time to end proceeding observation
                                        proceeding_obs.accessibility.left + proceeding_obs.min_duration)

                # Merge accessibility
                merged_accessibility = Interval(accessibility_start, accessibility_end) if accessibility_start <= accessibility_end else EmptyInterval()

        assert min_duration_req >= max(preceeding_obs.min_duration, proceeding_obs.min_duration),\
                "Calculated minimum duration requirement does not satisfy parent observation opportunities' minimum duration requirements."

        if merged_accessibility.left < 0.0 or merged_accessibility.right < 0.0:
            raise ValueError(f"Calculated merged accessibility has negative bounds: {merged_accessibility}. This should not happen.")

        # calculate maximum duration requirement
        max_duration_req = min(self.max_duration, other.max_duration)

        # Return Time Requirements
        return merged_accessibility, min_duration_req, max_duration_req
    
    # @staticmethod
    # def __merge_slew_angles(task_slew_angles : Dict[GenericObservationTask, Interval]) -> Interval:
    #     """ Calculates the joint slew angle requirements for the observation opportunity. """
    #     return reduce(lambda a, b: a.intersection(b), task_slew_angles.values())
        
    @staticmethod
    def __merge_task_availability(tasks : Set[GenericObservationTask]) -> Interval:
        """ Calculates the joint availability interval for the observation opportunity based on the availability intervals of its parent tasks. """
        task_availabilities : List[Interval] = [task.availability for task in tasks]
        return reduce(lambda a, b: a.intersection(b), task_availabilities)    

    def can_merge(self, 
                  other : 'ObservationOpportunity', 
                  max_duration : float = 5*60,
                  must_overlap : bool = False 
                ) -> bool:
        """ 
        Check if two tasks can be merged based on their time and slew angle. 
        
        **Arguments:**
        - `other`: Other task observation opportunity to merge with.
        - `max_duration`: The maximum allowed duration for the merged observation opportunity in seconds [s].
        """
               
        # Validate inputs
        assert isinstance(other, ObservationOpportunity), "The other observation opportunity must be an instance of `ObservationOpportunity`."
        assert isinstance(max_duration, (float, int)), "`max_duration` must be a number."
        assert max_duration > 0.0, "`max_duration` must be positive."
        
        # Calculate slew angles overlap
        slew_angles_overlap : bool \
            = self.slew_angles.overlaps(other.slew_angles)

        # Calculate accessibility overlap and duration requirements
        merged_accessibility, min_duration_req, max_duration_req \
            = self._calc_merged_time_reqs(other, must_overlap)

        # Gather joint observation targets
        # my_targets = set(self.get_location())
        # other_targets = set(other.get_location())
        # location_overlap : bool = len(my_targets.intersection(other_targets)) > 0

        # merge task accessibilities
        task_availabilities : List[Interval] = [task.availability for task in self.tasks.union(other.tasks)]
        merged_task_availability : Interval = reduce(lambda a, b: a.intersection(b), task_availabilities)
        
        # check if merged accessibility is within the intersection of the parent tasks' availability intervals
        tasks_are_available : bool = not merged_task_availability.is_empty()
        tasks_are_accessible : bool = tasks_are_available and merged_accessibility.is_subset(merged_task_availability)

        return (
            # not the same observation opportunity
            other is not self and
            # same instrument      
            self.instrument_name == other.instrument_name and
            # duration requirements do not exceed maximum allowed duration
            min_duration_req <= max_duration and
            # joint minimum duration requirements is valid
            not math.isnan(min_duration_req) and
            # accessibility window encompasses the duration requirements
            min_duration_req <= merged_accessibility.span() and
            # merged accessibility does not exceed maximum allowed duration
            merged_accessibility.span() <= max_duration_req and
            # slew angles overlap
            slew_angles_overlap and
            # there exist a valid joint accessibility window 
            not merged_accessibility.is_empty() and
            merged_accessibility.left >= 0.0 and
            merged_accessibility.right >= 0.0 and
            # there exists overlap between the tasks' availability intervals
            tasks_are_available and
            # there merged accessibility is within the intersection of the parent tasks' availability intervals
            tasks_are_accessible
        ) 
    
    def merge(self, 
              other : 'ObservationOpportunity', 
              must_overlap : bool = False, 
              keep_id : bool = True
            ) -> 'ObservationOpportunity':
        """ 
        Merge two task observation opportunities into one. 
                
        **Arguments:**
        - `other`: The other observation opportunity to merge with.
        - `must_overlap`: If True, the observation opportunities must overlap in accessibility to be merged.
        - `max_duration`: The maximum allowed duration for the merged observation opportunity in seconds [s] (default: 120 seconds = 2 minutes).
        """
         # Check other observation opportunity's type
        assert isinstance(other, ObservationOpportunity), \
            "can only merge with observation opportunities of type `ObservationOpportunity`."
        
        # Merge parent tasks
        merged_tasks = {task for task in self.tasks}
        merged_tasks.update({task for task in other.tasks})

        # Ensure instrument compatibility
        assert self.instrument_name == other.instrument_name, \
            "tasks pertain to observations with different instruments."

        # Merge task-specific time requirements
        task_accessibility, task_min_duration, task_slew_angles \
            = self.__compile_reqs(other)

        # Calculate accessibility overlap and duration requirements
        merged_accessibility, min_duration_req, max_duration_req \
            = self._calc_merged_time_reqs(other, must_overlap)

        # Calculate slew angles overlap
        merged_slew_angles : Interval \
            = self.slew_angles.intersection(other.slew_angles) 

        # Validate merged observation opportunity parameters
        assert not merged_accessibility.is_empty(), "joint observation opportunity availability is empty."
        assert not math.isnan(min_duration_req) , "minimum duration requirement is invalid."
        assert min_duration_req <= merged_accessibility.span(), "minimum duration requirements exceed accessibility span."
        assert not merged_slew_angles.is_empty(), "slew angles do not overlap."
    
        # Return merged observation opportunity
        return ObservationOpportunity(
            tasks = merged_tasks,
            instrument_name = self.instrument_name,
            task_accessibility=task_accessibility,
            task_slew_angles=task_slew_angles,
            task_min_duration=task_min_duration,
            accessibility = merged_accessibility,
            slew_angles = merged_slew_angles,
            min_duration = min_duration_req,
            max_duration = max_duration_req,
            id = self.id if keep_id else None
        )
    
    def __compile_reqs(self, 
                       other : 'ObservationOpportunity'
                      ) -> Tuple[dict, dict, dict]:
        """ 
        Compiles the time requirements for merging two observation opportunities. 
        
        **Arguments:**
        - `other`: The other observation opportunity to merge with.
        - `must_overlap`: If True, the observation opportunities must overlap in accessibility to be merged.
        """
        # compile all task accessibilities requirements
        task_accessibility = dict(self.task_accessibility)
        task_accessibility.update(other.task_accessibility)

        # compile all task duration requirements
        task_min_duration = dict(self.task_min_duration)
        task_min_duration.update(other.task_min_duration)

        # compile all observation slew angle requirements
        task_slew_angles = dict(self.task_slew_angles)
        task_slew_angles.update(other.task_slew_angles)

        # return merged time requirements
        return task_accessibility, task_min_duration, task_slew_angles

    # -------------------------------------------
    #               ID METHODS 
    # -------------------------------------------

    @staticmethod
    def __generate_id(tasks : Union[GenericObservationTask, set],
                    instrument_name : str,
                    accessibility : Interval,
                    ) -> str:
        """
        Deterministic ID for a task observation opportunity based on:
        - parent generic task IDs
        - access interval [start, end)
        - optional agent/instrument/index
        """
        # check inputs
        if isinstance(tasks, GenericObservationTask):
            tasks = {tasks}
        elif isinstance(tasks, list):
            tasks = set(tasks)
        assert isinstance(tasks, set), \
            "parent_tasks must be a set of `GenericObservationTask`"

        # Define namespace for UUID generation
        OBS_OPPORTUNITY_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

        # Collect parent IDs (sorted string representation)
        parent_part = ",".join(sorted([str(p.id) for p in tasks]))

        # Collect access interval [start, end]
        start, end = accessibility.left, accessibility.right
        interval_part = f"{np.round(start,3)}_{np.round(end,3)}"

        # Collect additional info for disambiguation
        extras = []
        extras.append(f"instrument={instrument_name}")
        extras_part = ";".join(extras)

        # Create canonical name string
        name = f"{parent_part}|{extras_part}|{interval_part}"

        # Return deterministic UUID derived from name
        return str(uuid.uuid5(OBS_OPPORTUNITY_NAMESPACE, name))    

    def regenerate_id(self) -> None:
        """ Resets the ID of the observation opportunity. """
        self.id = self.__generate_id(self.tasks, self.instrument_name, self.accessibility)
    
    # -------------------------------------------
    #               GETTERS
    # -------------------------------------------
    def get_location(self) -> List[tuple]:
        """ Collects the location information of all parent tasks. """
        return list({loc for task in self.tasks for loc in task.location})
    
    def get_objectives(self) -> List[MissionObjective]:
        """ Collects the objectives of all parent tasks. """
        return list({task.objective for task in self.tasks})
    
    def get_priority(self) -> float:
        """ Collects the priority of all parent tasks. """
        return sum(task.priority for task in self.tasks) if self.tasks else 0.0
    
    def get_earliest_task_start(self, task : GenericObservationTask, t : float = np.NINF) -> float:
        """ Returns the earliest start time of the observation opportunity for a given task. """        
        # the earliest start time is the maximum of the current time, 
        #  the observation opportunity's accessibility start, and the task's accessibility start
        access_left = self.accessibility.left
        task_left = self.task_accessibility[task.id].left

        if t < access_left:
            t = access_left
        if t < task_left:
            t = task_left

        return t
        
    
    def get_earliest_starts(self, t : float = np.NINF) -> Dict[GenericObservationTask, float]:
        """ Returns the earliest start time of each task being observed by this observation opportunity. """
        assert t in self.accessibility, \
            f"Time t={t} is outside the observation opportunity's accessibility interval {self.accessibility}."                    
        
        return {task : self.get_earliest_task_start(task, t) 
                for task in self.tasks}

    # -------------------------------------------
    #            UTILITY METHODS
    # -------------------------------------------
    def __repr__(self):
        return f"ObservationOpportunity_{self.id.split('-')[0]}"    
    
    def __hash__(self):
        return hash(self.id)
            
    def to_dict(self) -> dict:        
        # sort tasks by id for consistent serialization
        sorted_tasks = sorted(self.tasks, key=lambda task: task.id)

        # return dictionary representation
        return {
            "id": self.id,
            "tasks": [task.to_dict() for task in sorted_tasks],
            "accessibility": self.accessibility.to_dict(),
            "task_accessibility" : {task_id : interval.to_dict() for task_id, interval in self.task_accessibility.items()},
            "task_slew_angles" : {task_id : interval.to_dict() for task_id, interval in self.task_slew_angles.items()},
            "task_min_duration" : {task_id : duration for task_id, duration in self.task_min_duration.items()},
            "instrument_name": self.instrument_name,
            "max_duration" : self.max_duration,
            "min_duration": self.min_duration,
            "slew_angles": self.slew_angles.to_dict(),
            "availability" : self.availability.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationOpportunity":
        # reconstruct tasks from their dictionary representations
        tasks = {task_data['id'] : GenericObservationTask.from_dict(task_data) 
                 for task_data in d["tasks"]}

        # return reconstructed observation opportunity
        return ObservationOpportunity(
            id=d["id"],
            tasks = set(tasks.values()),
            instrument_name = d["instrument_name"],
            task_accessibility={task_id: Interval.from_dict(interval) 
                                  for task_id, interval in d["task_accessibility"].items()},
            task_slew_angles={task_id: Interval.from_dict(interval) 
                              for task_id, interval in d["task_slew_angles"].items()},
            task_min_duration={task_id: duration 
                               for task_id, duration in d["task_min_duration"].items()},
            accessibility = Interval.from_dict(d["accessibility"]),
            slew_angles = Interval.from_dict(d["slew_angles"]),
            min_duration = d["min_duration"],
            max_duration = d["max_duration"]
        )
 


class AtomicObservationOpportunity(ObservationOpportunity):
    def __init__(self, 
                 task : GenericObservationTask, 
                 instrument_name : str,
                 accessibility : Interval,
                 slew_angles : Interval,
                 min_duration : float,
                 max_duration : float = 15*60,
                 id : str = None
                ):
        """
        ## Atomic observation opportunity
        Represents an atomic observation opportunity that can be scheduled by an agent in order to fulfill a single generic observation task. 
        This is a special case of `ObservationOpportunity` where there is only one parent task.

        **Arguments:**
        - `task`: The parent generic observation task that this observation opportunity can fulfill.
        - `accessibility`: The time interval during which the observation opportunity is accessible for the task
        - `slew_angles`: The interval of slew angles [deg] available for the observation for the task.
        - `instrument_name`: The name of the instrument to be used for the observation.
        - `min_duration`: The minimum duration required for the observation in seconds [s] for the task.
        - `max_duration`: The maximum allowed duration for the observation opportunity in seconds [s] (default: 900 [s] = 15 [min]).
        - `id`: A unique identifier for the observation opportunity. If not provided, a new ID will be generated.
        """
        # validate inputs types
        assert isinstance(task, GenericObservationTask), "task must be an instance of `GenericObservationTask`."
        assert isinstance(instrument_name, str), "Instrument name must be a string."
        assert isinstance(accessibility, Interval), "Accessibility must be an `Interval` instance."
        assert isinstance(slew_angles, Interval), "Slew angles must be an `Interval` instance."
        assert isinstance(min_duration, (float, int)), "Minimum duration must be a number."
        assert isinstance(max_duration, (float, int)), "Maximum duration must be a number."               

        # initialize the observation opportunity using the parent task's requirements
        super().__init__(
            tasks = {task},
            instrument_name = instrument_name,
            task_accessibility = {task.id : accessibility},
            task_slew_angles = {task.id : slew_angles},
            task_min_duration = {task.id : min_duration},
            accessibility = accessibility,
            slew_angles = slew_angles,
            min_duration = min_duration,
            max_duration = max_duration,
            id = id
        )