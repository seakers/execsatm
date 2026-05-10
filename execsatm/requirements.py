from enum import Enum
from numbers import Number
import uuid
import numpy as np
from typing import Any, Dict, List, Set, Tuple, Union, Tuple
from pyparsing import ABC, abstractmethod

"""
---------------------------------
ABSTRACT REQUIREMENT DEFINITION
---------------------------------
"""
        
class RequirementTypes(Enum):
    CAPABILITY = 'capability'
    SPATIAL = 'spatial'
    PERFORMANCE = 'performance'
    SPECTRAL = 'spectral'

class MissionRequirement(ABC):
    def __init__(self, req_type : str, attribute: str, id : str = None):
        """
        ### Mission Requirement

        Initialize a mission requirement with a requirement type, attribute, strategy, and unique ID.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "temperature", "humidity").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # validate argument types
        assert isinstance(req_type, str), "Requirement type must be a string"
        assert isinstance(attribute, str), "Attribute must be a string"
        assert isinstance(id, str) or id is None, "ID must be a string or `None`"

        # validate argument values
        assert req_type.lower() in RequirementTypes._value2member_map_, f"Requirement type must be one of {list(RequirementTypes._value2member_map_.keys())}"

        # set attributes
        self.req_type : str = req_type.lower()
        self.attribute : str = attribute.lower()
        # TODO do we really need to enforce UUID format? Could the ID be attribute-dependent?
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())

    def calc_preference(self, attribute : str, value : Any) -> float:
        """Evaluates the preference function for a given parameter-value pair."""
        
        # check if attribute matches requirement attribute
        assert isinstance(attribute, str), "Attribute must be a string"
        assert attribute.lower() == self.attribute, \
            f"Attribute '{attribute}' does not match requirement attribute '{self.attribute}'"
        
        # calculate preference value
        result = self._eval_preference_function(value)

        # validate the result
        if not isinstance(result, Number):
            raise TypeError(f"Expected a numeric return value, got {type(result).__name__}")

        if not (0.0 - 1e-6 <= result <= 1.0 + 1e-6):  # allow small numerical tolerance
            raise ValueError(f"Return value {result} is not in [0, 1]")

        # Return the preference value
        return result
    
    @abstractmethod
    def _eval_preference_function(self, value : Any) -> float:
        """Evaluate the preference function for a given value."""

    @abstractmethod
    def __repr__(self):
        """String representation of the measurement requirement."""
        # return f"MissionRequirement(type={RequirementTypes._value2member_map_[self.req_type].name}, attribute={self.attribute})"
    
    def copy(self) -> 'MissionRequirement':
        """Create a copy of the measurement requirement."""
        return self.from_dict(self.to_dict())
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert the measurement requirement to a dictionary."""
        return {
            "req_type": self.req_type,
            "attribute": self.attribute,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a measurement requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        req_type = d.get("req_type")

        # initiate approriate requirement
        if req_type.lower() == RequirementTypes.PERFORMANCE.value:
            return PerformanceRequirement.from_dict(d)
        elif req_type.lower() == RequirementTypes.CAPABILITY.value:
            return CapabilityRequirement.from_dict(d)
        elif req_type.lower() == RequirementTypes.SPATIAL.value:
            return SpatialCoverageRequirement.from_dict(d)
        elif req_type.lower() == RequirementTypes.SPECTRAL.value:
            return SpectralRequirement.from_dict(d)

        raise NotImplementedError(f"Requirement type '{req_type}' not yet supported.")
    
    @abstractmethod
    def __eq__(self, other):
        """Check equality of two requirements."""
        assert isinstance(other, MissionRequirement), "Can only compare MissionRequirement instances"
        
        # return self.to_dict() == other.to_dict()
        comp_attrs = ['req_type', 'attribute', 'id']
        return all(getattr(self, attr) ==  getattr(other, attr) for attr in comp_attrs)
"""
------------------------------------
PERFORMANCE REQUIREMENT DEFINITIONS
------------------------------------
"""

class PerformancePreferenceStrategies(Enum):
    # Categorical
    CATEGORICAL = 'categorical'

    # Discrete
    DISCRETE = 'discrete'

    # No change
    CONSTANT = 'constant'
    
    # Higher val = better   
    EXP_SATURATION = 'exp_saturation'
    LOG_THRESHOLD = 'log_threshold'
    DEMINISHING_RETURNS = 'diminishing_returns'
    
    # Lower val = better
    EXP_DECAY = 'exp_decay'
    
    # Bounded
    GAUSSIAN = 'gaussian'
    TRIANGLE = 'triangle'
    
    # Interval Threshold-Based
    STEPS = 'discrete_steps'
    INTERVAL_INTERP = 'discrete_intervals'

class PerformanceRequirement(MissionRequirement):
    def __init__(self, 
                 attribute : str, 
                 strategy : str,
                 id = None):
        """
        ### Performance Requirement

        Initializes a generic measurement performance requirement
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.PERFORMANCE.value, attribute, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in PerformancePreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(PerformancePreferenceStrategies._value2member_map_.keys())}"
        
        # set attributes
        self.strategy : str = strategy.lower()

    def __repr__(self):
        """String representation of the measurement requirement."""
        return f"PerformanceRequirement(strategy={PerformancePreferenceStrategies._value2member_map_[self.strategy].name}, attribute={self.attribute})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a performance requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == PerformancePreferenceStrategies.CATEGORICAL.value:
            return CategoricalRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.CONSTANT.value:
            return ConstantValueRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.EXP_SATURATION.value:
            return ExpSaturationRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.LOG_THRESHOLD.value:
            return LogThresholdRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.EXP_DECAY.value:
            return ExpDecayRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.GAUSSIAN.value:
            return GaussianRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.TRIANGLE.value:
            return TriangleRequirement.from_dict(d)

        elif strategy == PerformancePreferenceStrategies.STEPS.value:
            return StepsRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.INTERVAL_INTERP.value:
            return IntervalInterpolationRequirement.from_dict(d)
        
        elif strategy == PerformancePreferenceStrategies.DEMINISHING_RETURNS.value:
            return DeminishingReturnsRequirement.from_dict(d)

        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

    @abstractmethod
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, PerformanceRequirement):
            return self.strategy == other.strategy
        return False
    
    @abstractmethod
    def to_dict(self):
        d = super().to_dict()
        d['strategy'] = self.strategy
        return d

class CategoricalRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 preferences : Dict[str, float],
                 id = None):
        """
        ### Categorical Requirement

        Initializes a requirement that assigns preference scores to categorical values.
        - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
        - :`preferences`: A dictionary mapping categorical values (strings) to preference scores (floats) in the range [0, 1].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.CATEGORICAL.value, id)
        
        # validate inputs
        assert isinstance(preferences, dict), "Preferences must be a dictionary"
        for key, val in preferences.items():
            assert isinstance(key, str), "Preference keys must be strings"
            assert isinstance(val, (int, float)), "Preference values must be numeric"
            assert 0.0 <= val <= 1.0, "Preference values must be in [0, 1]"
        
        # set attributes
        self.preferences : Dict[str, float] = {key.lower(): val for key,val in preferences.items()}
    
    def _eval_preference_function(self, value : str) -> float:
        # validate inputs
        assert isinstance(value, str), "Input value must be a string"

        # normalize value to lowercase string
        value = str(value).lower()

        # return preference value
        return self.preferences.get(value, 0.0) # default preference is 0.0 if category not found
        
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'CategoricalRequirement':
        """Create a categorical requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['attribute', 'strategy', 'preferences']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.CATEGORICAL.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.CATEGORICAL.value}'"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        preferences = dict.get("preferences")
        id = dict.get("id", None)
        
        # initiate requirement
        return cls(attribute, preferences, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, CategoricalRequirement):
            for key,value in self.preferences.items():
                if key not in other.preferences or other.preferences[key] != value:
                    return False
            return True
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['preferences'] = dict(self.preferences)
        return d
    
class ConstantValueRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,
                 value : float = 1.0,
                 id = None
                ):
        """
        ### Constant Value Requirement

        Initializes a requirement that always returns the same preference score.
        - :`attribute`: The attribute being measured.
        - :`value`: The constant preference score to return (default is 1.0).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.CONSTANT.value, id)

        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"
        assert 0.0 <= value <= 1.0, "Value must be in [0, 1]"

        # set attributes
        self.value : float = value

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Input value must be numeric"

        # return preference value
        return self.value # always returns the constant preference value
       
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ConstantValueRequirement':
        """Create a constant value requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.CONSTANT.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.CONSTANT.value}'"

        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        value = dict.get("value", 1.0)  # default to 1.0 if not provided
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, value, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, ConstantValueRequirement):
            return self.value == other.value
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['value'] = self.value
        return d
    
class ExpSaturationRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 sat_rate : float,
                 id = None
                ):
        """
        ### Exponential Saturation Requirement

        Initializes a requirement that uses an exponential saturation preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`sat_rate`: The rate at which preference saturates (higher values lead to quicker saturation). Must be non-negative.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.EXP_SATURATION.value, id)
        
        # validate inputs
        assert isinstance(sat_rate, (int, float)), "Saturation rate must be a number"
        assert sat_rate >= 0, "Saturation rate must be non-negative"

        # set attributes
        self.sat_rate : float = sat_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Evaluated value must be a number"
        assert value >= 0, "Evaluated value must be non-negative"

        # return preference value
        return 1.0 - np.exp(-self.sat_rate * value)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", sat_rate={self.sat_rate})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpSaturationRequirement':
        """Create an exponential saturation requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy', 'sat_rate']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.EXP_SATURATION.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.EXP_SATURATION.value}'"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        sat_rate = dict.get("sat_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, sat_rate, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, ExpSaturationRequirement):
            return abs(self.sat_rate - other.sat_rate) < 1e-6
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['sat_rate'] = self.sat_rate
        return d
    
class LogThresholdRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 slope : float, 
                 threshold : float, 
                 id = None
                ):
        """
        ### Logarithmic Threshold Requirement
        
        Initializes a requirement that uses a logarithmic threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`slope`: The slope of the logarithmic function (higher values lead to steeper transitions). Must be positive.
        - :`threshold`: The threshold value at which preference value is 0.5. Must be non-negative.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.LOG_THRESHOLD.value, id)
        
        # validate inputs
        assert isinstance(slope, (int, float)), "Slope must be a number"
        assert slope > 0, "Slope must be positive"
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        assert threshold >= 0, "Threshold must be non-negative"

        # set attributes
        self.slope : float = slope
        self.threshold : float = threshold
    
    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be a number"
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return 1 / (1 + np.exp(-self.slope * (value - self.threshold)))
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", slope={self.slope}, threshold={self.threshold})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'LogThresholdRequirement':
        """Create a log threshold requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'slope', 'threshold']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.LOG_THRESHOLD.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.LOG_THRESHOLD.value}'"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        slope = dict.get("slope")
        threshold = dict.get("threshold")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, slope, threshold, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, LogThresholdRequirement):
            return (abs(self.slope - other.slope) < 1e-6 and
                    abs(self.threshold - other.threshold) < 1e-6)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['slope'] = self.slope
        d['threshold'] = self.threshold
        return d

class DeminishingReturnsRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 slope : float, 
                 threshold : float, 
                 id = None
                ):
        """
        ### Diminishing Returns Requirement
        
        Initializes a requirement that uses the derivative of a logarithmic threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`slope`: The slope of the logarithmic function (higher values lead to steeper transitions). Must be positive.
        - :`threshold`: The threshold value at which preference value is 0.5. Must be non-negative.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.DEMINISHING_RETURNS.value, id)
        
        # validate inputs
        assert isinstance(slope, (int, float)), "Slope must be a number"
        assert slope > 0, "Slope must be positive"
        assert isinstance(threshold, (int, float)), "Threshold must be a number"
        assert threshold >= 0, "Threshold must be non-negative"

        # set attributes
        self.slope : float = slope
        self.threshold : float = threshold
    
    def _eval_preference_function(self, value : int) -> float:
        # validate inputs
        assert isinstance(value, int) and value > 0, \
            "Value must be a positive integer"
        
        # calculate preference values of value and value-1
        p_i_mins_1  = 1 / (1 + np.exp(-self.slope * (value - 1 - self.threshold)))
        p_i = 1 / (1 + np.exp(-self.slope * (value - self.threshold)))
    
        # return preference value
        return max(0.0, p_i - p_i_mins_1)

    def __repr__(self):
        return super().__repr__()[:-1] + f", slope={self.slope}, threshold={self.threshold})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'DeminishingReturnsRequirement':
        """Create a diminishing returns requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'slope', 'threshold']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.DEMINISHING_RETURNS.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.DEMINISHING_RETURNS.value}'"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        slope = dict.get("slope")
        threshold = dict.get("threshold")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, slope, threshold, id)
    
    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, DeminishingReturnsRequirement) and \
                (abs(self.slope - other.slope) < 1e-6 
                 and abs(self.threshold - other.threshold) < 1e-6)    
    
    def to_dict(self):
        d = super().to_dict()
        d['slope'] = self.slope
        d['threshold'] = self.threshold
        return d

class ExpDecayRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 decay_rate : float, 
                 id = None
                ):
        """
        ### Exponential Decay Requirement

        Initializes a requirement that uses an exponential decay preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`decay_rate`: The rate at which preference decays (higher values lead to quicker decay). Must be non-negative. 
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.EXP_DECAY.value, id)
        
        # validate inputs
        assert isinstance(decay_rate, (int, float)), "Decay rate must be a number"
        assert decay_rate >= 0, "Decay rate must be non-negative"
        
        # set attributes
        self.decay_rate : float = decay_rate

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be a number"
        assert value >= 0, "Value must be non-negative"
        
        # return preference value
        return np.exp(-self.decay_rate * value)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", decay_rate={self.decay_rate})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'ExpDecayRequirement':
        """Create an exponential decay requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'decay_rate']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.EXP_DECAY.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.EXP_DECAY.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        decay_rate = dict.get("decay_rate")
        id = dict.get("id", None) 

        # initiate requirement
        return cls(attribute, decay_rate, id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, ExpDecayRequirement):
            return abs(self.decay_rate - other.decay_rate) < 1e-6
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['decay_rate'] = self.decay_rate
        return d

class GaussianRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str,  
                 mean : float,
                 stddev : float,
                 id = None):
        """
        ### Gaussian Requirement

        Initializes a requirement that uses a Gaussian distribution as a threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`mean`: The mean value of the Gaussian function.
        - :`stddev`: The standard deviation of the Gaussian function. Must be positive.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.GAUSSIAN.value, id)

        # validate inputs
        assert isinstance(mean, (int, float)), "Average must be a number"
        assert isinstance(stddev, (int, float)), "Standard deviation must be a number"
        assert stddev > 0, "Standard deviation must be positive"

        # set attributes
        self.mean : float = mean
        self.stddev : float = stddev

    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Number of observations must be a number"
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return np.exp(-0.5 * ((value - self.mean) / self.stddev) ** 2)
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", mean={self.mean}, stddev={self.stddev})"

    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'GaussianRequirement':
        """Create a Gaussian requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'mean', 'stddev']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.GAUSSIAN.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.GAUSSIAN.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        mean = dict.get("mean")
        stddev = dict.get("stddev")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, mean, stddev, id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, GaussianRequirement):
            return (abs(self.mean - other.mean) < 1e-6 and
                    abs(self.stddev - other.stddev) < 1e-6)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['mean'] = self.mean
        d['stddev'] = self.stddev
        return d

class TriangleRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 reference : float,
                 width : float, 
                 id = None):
        """
        ### Triangle Requirement

        Initializes a requirement that uses a triangular threshold preference function.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`reference`: The reference value at which preference is maximized.
        - :`width`: The width of the triangle base (preference drops to 0.0 at reference ± width / 2). Must be positive.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.TRIANGLE.value, id)

        # validate inputs
        assert isinstance(reference, (int, float)), "Reference must be a number"
        assert isinstance(width, (int, float)), "Width must be a number"
        assert width > 0, "Width must be positive"
        
        # set attributes
        self.reference : float = reference
        self.width : float = width
    
    def _eval_preference_function(self, value : float) -> float:
        # validate inputs
        assert isinstance(value, (int, float)), "Number of observations must be a number"
        assert value >= 0, "Number of observations must be non-negative"

        # return preference value
        return max(0.0, 1.0 - abs(value - self.reference) / (self.width / 2))

    def __repr__(self):
        return super().__repr__()[:-1] + f", reference={self.reference}, width={self.width})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'TriangleRequirement':
        """Create a triangle requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'reference', 'width']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert dict.get("strategy").lower() == PerformancePreferenceStrategies.TRIANGLE.value, \
            f"Strategy does not match requirement definition. Must be '{PerformancePreferenceStrategies.TRIANGLE.value}'"
        
        # unpack dictionary
        req_type = dict.get("req_type")
        attribute = dict.get("attribute")
        reference = dict.get("reference")
        width = dict.get("width")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, reference, width, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, TriangleRequirement):
            return (abs(self.reference - other.reference) < 1e-6 and
                    abs(self.width - other.width) < 1e-6)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['reference'] = self.reference
        d['width'] = self.width
        return d

class StepsRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 thresholds : List[float],
                 scores : List[float],
                 id = None):
        """
        ### Discrete Steps Requirement
        
        Initializes a requirement that uses discrete step functions for preference evaluation.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`thresholds`: A list of numeric thresholds defining the steps (must be in ascending order).
        - :`scores`: A list of preference scores corresponding to each threshold interval (must be in [0, 1]).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """
        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.STEPS.value, id)
        
        # validate inputs
        assert isinstance(thresholds, list), "Thresholds must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(thresholds) + 1 == len(scores), \
            "Scores must have the same length as thresholds plus one"
        for threshold in thresholds:
            assert isinstance(threshold, (int, float)), "Thresholds must be numeric"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), "All values in `thresholds` must be ascending."
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"

        # set attributes
        self.thresholds = [threshold for threshold in thresholds]
        self.scores = [score for score in scores] # assumes scores match thresholds in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # return preference value based on discrete levels
        for threshold,score in zip(self.thresholds,self.scores[:-1]):
            if value < threshold:
                return score
                    
        if self.thresholds[-1] <= value:
            return self.scores[-1] 

        # fallback; should not reach here
        raise ValueError("Value does not fall within any defined thresholds.")    

    def __repr__(self):
        return super().__repr__()[:-1] + f", thresholds={self.thresholds}, scores={self.scores})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'StepsRequirement':
        """Create a discrete levels requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'thresholds', 'scores']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, thresholds, scores, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, StepsRequirement):
            if len(self.thresholds) != len(other.thresholds):
                return False
            if len(self.scores) != len(other.scores):
                return False

            matches_thresholds = all(abs(a - b) < 1e-6 for a, b in zip(self.thresholds, other.thresholds))
            matches_scores = all(abs(a - b) < 1e-6 for a, b in zip(self.scores, other.scores))

            return matches_thresholds and matches_scores
        
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['thresholds'] = list(self.thresholds)
        d['scores'] = list(self.scores)
        return d

class IntervalInterpolationRequirement(PerformanceRequirement):
    def __init__(self, 
                 attribute : str, 
                 thresholds : List[float],
                 scores : List[float],
                 id = None):
        """
        ### Interval Interpolation Requirement

        Initializes a requirement that uses interval-based linear interpolation for preference evaluation.
        - :`req_type`: The type of requirement (e.g., "capability", "temporal", "spatial").
        - :`attribute`: The attribute being measured (e.g., "data collected", "observations made").
        - :`thresholds`: A list of numeric thresholds defining the breakpoints (must be in ascending order).
        - :`scores`: A list of preference scores corresponding to each threshold (must be in [0, 1] and same length as thresholds).
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated
        """

        # initiate parent class
        super().__init__(attribute, PerformancePreferenceStrategies.INTERVAL_INTERP.value, id)
        
        # validate inputs
        assert isinstance(thresholds, list), "Intervals must be a list"
        assert isinstance(scores, list), "Scores must be a list"
        assert len(thresholds) == len(scores), "Intervals and scores must have the same length"
        for interval in thresholds:
            assert isinstance(interval, (int, float)), "Intervals must be numeric"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), \
            "All values in `intervals` must be ascending."
        for score in scores:
            assert isinstance(score, (int, float)), "Scores must be numeric"
            assert 0.0 <= score <= 1.0, "Scores must be in [0, 1]"

        # set attributes
        self.thresholds = [threshold for threshold in thresholds]
        self.scores = [score for score in scores] # assumes scores match intervals in length and order

    def _eval_preference_function(self, value):
        # validate inputs
        assert isinstance(value, (int, float)), "Value must be numeric"

        # find if value is between two intervals and interpolate score
        if self.thresholds[-1] < value:
            return self.scores[-1]

        # initialize previous values
        prev_threshold,prev_score = np.NINF, self.scores[0]

        # iterate through intervals
        for threshold,score in zip(self.thresholds,self.scores):
            # check if value is within current interval
            if prev_threshold < value <= threshold:
                # do not interpolate if previous threshold is -inf
                if prev_threshold == np.NINF: return score
                
                # linear interpolation
                m = (score - prev_score) / (threshold - prev_threshold) # slope
                return prev_score + m * (value - prev_threshold)        # interpolated score
            
            # update previous values for next interval
            prev_threshold,prev_score = threshold, score
        
        # fallback; should not reach here
        raise ValueError("Value does not fall within any defined intervals.")
        
    def __repr__(self):
        return super().__repr__()[:-1] + f", thresholds={self.thresholds}, scores={self.scores})"
    
    @classmethod
    def from_dict(cls, dict: Dict[str, Union[str, float]]) -> 'IntervalInterpolationRequirement':
        """Create a discrete intervals requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'thresholds', 'scores']
        assert all(key in dict for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        
        # unpack dictionary
        attribute = dict.get("attribute")
        thresholds = dict.get("thresholds")
        scores = dict.get("scores")
        id = dict.get("id", None)

        # initiate requirement
        return cls(attribute, thresholds, scores, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, IntervalInterpolationRequirement):
            if len(self.thresholds) != len(other.thresholds):
                return False
            if len(self.scores) != len(other.scores):
                return False

            matches_thresholds = all(abs(a - b) < 1e-6 for a, b in zip(self.thresholds, other.thresholds))
            matches_scores = all(abs(a - b) < 1e-6 for a, b in zip(self.scores, other.scores))

            return matches_thresholds and matches_scores
        
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['thresholds'] = list(self.thresholds)
        d['scores'] = list(self.scores)
        return d

"""
-----------------------------
CAPABILITY REQUIREMENT DEFINITIONS
-----------------------------
"""
class CapabilityPreferenceStrategies(Enum):
    # Explicit categorical matching
    EXPLICIT = 'explicit'

class CapabilityRequirement(MissionRequirement):
    def __init__(self, 
                 attribute : str, 
                 strategy : str,
                 id = None):
        """
        ### Capability Requirement

        Initializes a generic measurement capability requirement
        - :`attribute`: The attribute being evaluated (e.g., "instrument capability").
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.CAPABILITY.value, attribute, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in CapabilityPreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(CapabilityPreferenceStrategies._value2member_map_.keys())}"
        
        # set attributes
        self.strategy : str = strategy.lower()

    def __repr__(self):
        """String representation of the capability requirement."""
        return f"CapabilityRequirement(strategy={CapabilityPreferenceStrategies._value2member_map_[self.strategy].name}, attribute={self.attribute})"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a capability requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"    
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == CapabilityPreferenceStrategies.EXPLICIT.value:
            return ExplicitCapabilityRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

    def __eq__(self, other : 'MissionRequirement') -> bool:
        if super().__eq__(other) and isinstance(other, CapabilityRequirement):
            return self.strategy == other.strategy
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d["strategy"] = self.strategy
        return d

class ExplicitCapabilityRequirement(CapabilityRequirement):
    def __init__(self, 
                 attribute : str, 
                 valid_values : Union[List[str], Set[str]],
                 id = None):
        """
        ### Explicit Capability Requirement

        Initializes a requirement that accepts any value from a predefined set of valid categorical values.
        - :`attribute`: The attribute being measured (e.g., instrument type, agent type, etc.).
        - :`valid_values`: A set of valid categorical values (strings) that are acceptable.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(attribute, CapabilityPreferenceStrategies.EXPLICIT.value, id)

        # validate inputs
        assert isinstance(valid_values, (list, set)), "Valid values must be a list or set"
        assert all(isinstance(val, str) for val in valid_values), "All valid values must be strings"

        # set attributes
        self.valid_values : Set[str] = {val.lower() for val in valid_values}

    def _eval_preference_function(self, value : str) -> float:
        """Evaluate the preference function for a given capability value."""
        
        # validate inputs
        assert isinstance(value, str), "Input value must be a string"

        # normalize value to lowercase string
        value = str(value).lower()  

        # return preference value
        return 1.0 if value in self.valid_values else 0.0
    
    def to_dict(self):
        d = super().to_dict()
        d["valid_values"] = sorted(self.valid_values)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'ExplicitCapabilityRequirement':
        """Create an explicit capability requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'attribute', 'valid_values']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy") == CapabilityPreferenceStrategies.EXPLICIT.value, \
            f"Strategy does not match requirement definition. Must be '{CapabilityPreferenceStrategies.EXPLICIT.value}'"
        
        # unpack dictionary
        attribute = d.get("attribute")
        valid_values : list = d.get("valid_values")
        id = d.get("id", None)

        # initiate requirement
        return cls(attribute, valid_values, id)
    
    def __eq__(self, other):
        return super().__eq__(other) \
            and isinstance(other, ExplicitCapabilityRequirement) and \
               self.valid_values == other.valid_values

"""
---------------------------------
SPATIAL REQUIREMENT DEFINITIONS
---------------------------------
"""
class SpatialPreferenceStrategies(Enum):
    SINGLE_POINT = 'single_point'
    MULTI_POINT = 'multi_point'
    GRID = 'grid'

class SpatialCoverageRequirement(MissionRequirement):
    ATTRIBUTE = 'location'

    def __init__(self, 
                 strategy : str,
                 id = None):
        """
        ### Spatial Coverage Requirement

        Initializes a generic coverage requirement.
        - :`strategy`: Name of the preference function strategy to be used (e.g., "categorical", "exp_saturation").
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(RequirementTypes.SPATIAL.value, self.ATTRIBUTE, id)

        # validate inputs
        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in SpatialPreferenceStrategies._value2member_map_, f"Preference strategy must be one of {list(SpatialPreferenceStrategies._value2member_map_.keys())}"

        # set attributes
        self.strategy : str = strategy.lower()

    def haversine_np(self, lat1 : float, lon1 : float, lat2 : float, lon2 : float) -> float:
        """
        Calculate the great circle distance between two points on the earth in [km] (specified in decimal degrees)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Calculate angular difference in radians
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Haversine formula
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        # Calculate the arc distance
        c = 2 * np.arcsin(np.sqrt(a))

        # Return great circle distance in kilometers
        return 6378.137 * c

    def __repr__(self):
        """String representation of the coverage requirement."""
        return f"SpatialRequirement(strategy={SpatialPreferenceStrategies._value2member_map_[self.strategy].name})"
      
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MissionRequirement':
        """Create a spatial requirement from a dictionary."""
        
        # validate input dictionary
        required_keys = ['req_type', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"    
        # unpack dictionary
        strategy = d.get("strategy").lower()

        # initiate approriate requirement 
        if strategy == SpatialPreferenceStrategies.SINGLE_POINT.value:
            return SinglePointSpatialRequirement.from_dict(d)
        elif strategy == SpatialPreferenceStrategies.MULTI_POINT.value:
            return MultiPointSpatialRequirement.from_dict(d)
        elif strategy == SpatialPreferenceStrategies.GRID.value:
            return GridSpatialRequirement.from_dict(d)
        
        # Additional strategies can be implemented here
        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")
    
    @abstractmethod
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, SpatialCoverageRequirement):
            return self.strategy == other.strategy
        return False
    
    @abstractmethod
    def to_dict(self):
        d = super().to_dict()
        d["strategy"] = self.strategy
        return d

class SinglePointSpatialRequirement(SpatialCoverageRequirement):
    def __init__(self, 
                 target : Union[Tuple, list],
                 distance_threshold : float,
                 id = None):
        """
        ### Single Point Spatial Requirement

        Initializes a requirement that evaluates preference based on proximity to a single target point.
        - :`target_point`: A tuple representing the target location as (latitude [deg], longitude [deg], grid idx, gp idx).
        - :`distance_threshold`: The distance threshold for full preference in [km].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.SINGLE_POINT.value, id)

        # validate inputs
        if isinstance(target, list):
            if len(target) == 1:
                target = target[0]
            else:
                raise ValueError("Target must be a single tuple of (latitude, longitude, grid idx, gp idx)")
            
        assert isinstance(target, tuple) and len(target) == 4, "Target point must be a tuple of (latitude, longitude)"
        lat, lon, grid_idx, gp_idx = target
        assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
        assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
        assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
        assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
        assert grid_idx >= 0, "Grid index must be non-negative"
        assert gp_idx >= 0, "GP index must be non-negative"
        assert isinstance(distance_threshold, (int, float)), "Distance threshold must be numeric"
        assert distance_threshold >= 0, "Distance threshold must be non-negative"

        # set attributes
        self.target : Tuple[float, float, int, int] = target
        self.distance_threshold : float = distance_threshold

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, list):
            if len(location) == 1:
                location = location[0]
            else:
                raise ValueError("Location must be a single tuple of (latitude, longitude, grid idx, gp idx)")

        assert isinstance(location, tuple) and len(location) == 4, "Location must be a tuple of (latitude, longitude, grid idx, gp idx)"
        lat, lon, grid_idx, gp_idx = location
        assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
        assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
        assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
        assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
        assert grid_idx >= 0, "Grid index must be non-negative"
        assert gp_idx >= 0, "GP index must be non-negative"

        # check for exact match
        if location == self.target: return 1.0

        # calculate distance to target point
        target_lat, target_lon, _, _ = self.target
        distance = self.haversine_np(lat, lon, target_lat, target_lon)

        # return preference value based on distance threshold
        return float(distance <= self.distance_threshold)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'SinglePointSpatialRequirement':
        """Create a single point spatial requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'target', 'distance_threshold']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.SINGLE_POINT.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.SINGLE_POINT.value}'"
        
        # unpack dictionary
        target = d.get("target")
        distance_threshold = d.get("distance_threshold")
        id = d.get("id", None)

        # initiate requirement
        return cls(target, distance_threshold, id)
    
    def __repr__(self):
        """String representation of the single point spatial requirement."""
        return super().__repr__()[:-1] + f", target={self.target[2],self.target[3]})"
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, SinglePointSpatialRequirement):
            matchin_targets = all(abs(a - b) < 1e-6 for a, b in zip(self.target, other.target))
            return (matchin_targets and abs(self.distance_threshold - other.distance_threshold) < 1e-6)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['target'] = tuple(self.target)
        d['distance_threshold'] = self.distance_threshold
        return d
    
class MultiPointSpatialRequirement(SpatialCoverageRequirement):
    def __init__(self, 
                 targets : List[Tuple[float, float, int, int]],
                 distance_threshold : float,
                 id = None):
        """
        ### Multi Point Target Spatial Requirement

        Initializes a requirement that evaluates preference based on proximity to a list of target points.
        - :`targets`: A list of tuples representing target locations as (latitude [deg], longitude [deg], grid idx, gp idx).
        - :`distance_threshold`: The distance threshold for full preference in [km].
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.MULTI_POINT.value, id)

        # validate inputs
        assert isinstance(targets, list) and len(targets) > 0, "Target list must be a non-empty list of target points"
        for target in targets:
            assert isinstance(target, tuple) and len(target) == 4, "Each target point must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = target
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"
        assert isinstance(distance_threshold, (int, float)), "Distance threshold must be numeric"
        assert distance_threshold >= 0, "Distance threshold must be non-negative"

        # set attributes
        self.targets : List[Tuple[float, float, int, int]] = targets
        self.distance_threshold : float = distance_threshold

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, tuple):
            if len(location) == 4:
                location = [location]
            else:
                raise ValueError("Location must be a list of tuples of (latitude, longitude, grid idx, gp idx)")

        for loc in location:
            assert isinstance(loc, tuple) and len(loc) == 4, "Location must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = loc
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), "Grid index and GP index must be integers"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"

        # check for exact match with any target
        for target in self.targets:
            if location == target:
                return 1.0

        # calculate distances to all target points
        for target in self.targets:
            target_lat, target_lon, _, _ = target
            distance = self.haversine_np(lat, lon, target_lat, target_lon)
            if distance <= self.distance_threshold:
                return 1.0

        # return preference value based on distance threshold
        return 0.0
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", num_targets={len(self.targets)})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'MultiPointSpatialRequirement':
        """Create a target list spatial requirement from a dictionary."""

        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'targets', 'distance_threshold']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.MULTI_POINT.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.MULTI_POINT.value}'"

        # unpack dictionary
        targets = d.get("targets")
        distance_threshold = d.get("distance_threshold")
        id = d.get("id", None)
        
        # initiate requirement
        return cls(targets, distance_threshold, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, MultiPointSpatialRequirement):
            if len(self.targets) != len(other.targets):
                return False
            matching_targets = all(
                any(all(abs(a - b) < 1e-6 for a, b in zip(self_target, other_target)) 
                    for other_target in other.targets)
                for self_target in self.targets
            )
            return (matching_targets and abs(self.distance_threshold - other.distance_threshold) < 1e-6)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['targets'] = [tuple(target) for target in self.targets]
        d['distance_threshold'] = self.distance_threshold
        return d

class GridSpatialRequirement(SpatialCoverageRequirement):
    # TODO load grid definitions from file or external source and evaluate accordingly

    def __init__(self, 
                 grid_name : str,
                 grid_index : int,
                 grid_size : int,
                 id = None):
        """
        ### Grid Coverage Spatial Requirement

        Initializes a requirement that evaluates preference based on coverage of specified grid cells.
        - :`grid_name`: The name of the grid (e.g., "global", "regional").
        - :`grid_index`: The index of the grid cell.
        - :`grid_size`: The size of the grid cell in degrees.
        - :`id`: Optional unique identifier for the requirement. If not provided, a UUID will be generated.
        """

        # initiate parent class
        super().__init__(SpatialPreferenceStrategies.GRID.value, id)

        # validate inputs
        assert isinstance(grid_name, str), "Grid name must be a string"
        assert isinstance(grid_index, int) and grid_index >= 0, "Grid index must be a non-negative integer"
        assert isinstance(grid_size, int) and grid_size > 0, "Grid size must be a positive integer"
        
        # set attributes
        self.grid_name : str = grid_name
        self.grid_index : int = grid_index
        self.grid_size : int = grid_size

    def _eval_preference_function(self, location : Union[Tuple, list]) -> float:
        """Evaluate the preference function for a given location."""
        
        # validate inputs
        if isinstance(location, tuple):
            if len(location) == 4:
                location = [location]
            else:
                raise ValueError("Location must be a list of tuples of (latitude, longitude, grid idx, gp idx)")

        for loc in location:
            assert isinstance(loc, tuple) and len(loc) == 4, "Locations must be a tuple of (latitude, longitude, grid idx, gp idx)"
            lat, lon, grid_idx, gp_idx = loc
            assert isinstance(lat, (int, float)) and isinstance(lon, (int, float)), "Latitude and longitude must be numeric"
            assert -90.0 <= lat <= 90.0, "Latitude must be in [-90, 90]"
            assert -180.0 <= lon <= 180.0, "Longitude must be in [-180, 180]"
            assert isinstance(grid_idx, int) and isinstance(gp_idx, int), \
                "Grid index and GP index must be integers"
            assert grid_idx >= 0, "Grid index must be non-negative"
            assert gp_idx >= 0, "GP index must be non-negative"

            # return preference value based on grid index match
            return 1.0 if grid_idx == self.grid_index and gp_idx < self.grid_size else 0.0        
        
        return 0.0
    
    def __repr__(self):
        return super().__repr__()[:-1] + f", grid_name={self.grid_name}, grid_index={self.grid_index}, grid_size={self.grid_size})"

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, float]]) -> 'GridSpatialRequirement':
        """Create a grid coverage spatial requirement from a dictionary."""
        
        # validate input dictionary
        required_keys = ['req_type', 'strategy', 'grid_name', 'grid_index', 'grid_size']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpatialPreferenceStrategies.GRID.value, \
            f"Strategy does not match requirement definition. Must be '{SpatialPreferenceStrategies.GRID.value}'"  
        
        # unpack dictionary
        grid_name = d.get("grid_name")
        grid_index = d.get("grid_index")
        grid_size = d.get("grid_size")
        id = d.get("id", None)  

        # initiate requirement
        return cls(grid_name, grid_index, grid_size, id)
    
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, GridSpatialRequirement):
            return (self.grid_name == other.grid_name and
                    self.grid_index == other.grid_index and
                    self.grid_size == other.grid_size)
        return False
    
    def to_dict(self):
        d = super().to_dict()
        d['grid_name'] = self.grid_name
        d['grid_index'] = self.grid_index
        d['grid_size'] = self.grid_size
        return d
    
"""
---------------------------------
SPECTRAL REQUIREMENT DEFINITIONS
---------------------------------
"""
# All spectral requirements accept a List[Tuple[float, float, float]] as input value,
#   where each tuple is (center_nm, bandwidth_nm, resolution_nm).

class SpectralPreferenceStrategies(Enum):
    BAND_COUNT  = 'band_count'
    RESOLUTION  = 'spectral_resolution'
    RANGE       = 'spectral_range'
    TIERED      = 'tiered'

class SpectralRequirement(MissionRequirement):
    ATTRIBUTE = 'spectral_bands'

    def __init__(self, strategy: str, id=None):
        """
        ### Spectral Requirement

        Base class for spectral requirements. All subclasses receive the instrument's band list
        as `value`: `List[Tuple[float, float, float]]` = `[(center_nm, bandwidth_nm, resolution_nm), ...]`.
        - :`strategy`: Name of the spectral preference strategy.
        - :`id`: Optional unique identifier.
        """
        super().__init__(RequirementTypes.SPECTRAL.value, self.ATTRIBUTE, id)

        assert isinstance(strategy, str), "Preference strategy must be a string"
        assert strategy.lower() in SpectralPreferenceStrategies._value2member_map_, \
            f"Preference strategy must be one of {list(SpectralPreferenceStrategies._value2member_map_.keys())}"

        self.strategy: str = strategy.lower()

    def _validate_bands(self, bands: List[Tuple]) -> None:
        assert isinstance(bands, list), "Bands must be a list"
        for band in bands:
            assert isinstance(band, (tuple, list)) and len(band) == 3, \
                "Each band must be a tuple of (center_nm, bandwidth_nm, resolution_nm)"
            center, bw, res = band
            assert isinstance(center, (int, float)) and center > 0, "Center wavelength must be a positive number"
            assert isinstance(bw, (int, float)) and bw > 0, "Bandwidth must be a positive number"
            assert isinstance(res, (int, float)) and res > 0, "Resolution must be a positive number"

    def _filter_bands(self, bands: List[Tuple], wavelength_range: Tuple[float, float]) -> List[Tuple]:
        """Return bands whose center wavelength falls within [min_nm, max_nm]."""
        if wavelength_range is None:
            return bands
        min_nm, max_nm = wavelength_range
        return [b for b in bands if min_nm <= b[0] <= max_nm]

    def __repr__(self):
        return f"SpectralRequirement(strategy={SpectralPreferenceStrategies._value2member_map_[self.strategy].name})"

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpectralRequirement':
        """Create a spectral requirement from a dictionary."""
        required_keys = ['req_type', 'strategy']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        strategy = d.get("strategy").lower()

        if strategy == SpectralPreferenceStrategies.BAND_COUNT.value:
            return SpectralBandCountRequirement.from_dict(d)
        elif strategy == SpectralPreferenceStrategies.RESOLUTION.value:
            return SpectralResolutionRequirement.from_dict(d)
        elif strategy == SpectralPreferenceStrategies.RANGE.value:
            return SpectralRangeRequirement.from_dict(d)
        elif strategy == SpectralPreferenceStrategies.TIERED.value:
            return TieredSpectralRequirement.from_dict(d)

        raise NotImplementedError(f"Preference function for strategy '{strategy}' not yet supported.")

    @abstractmethod
    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, SpectralRequirement):
            return self.strategy == other.strategy
        return False

    @abstractmethod
    def to_dict(self):
        d = super().to_dict()
        d["strategy"] = self.strategy
        return d


class SpectralBandCountRequirement(SpectralRequirement):
    def __init__(self,
                 wavelength_range: Tuple[float, float],
                 thresholds: List[int],
                 scores: List[float],
                 id=None):
        """
        ### Spectral Band Count Requirement

        Evaluates preference based on the number of bands whose center wavelength falls
        within `wavelength_range`. Scoring follows StepsRequirement semantics:
        `scores[i]` is returned when the band count is less than `thresholds[i]`.
        - :`wavelength_range`: `(min_nm, max_nm)` filter, or `None` to count all bands.
        - :`thresholds`: Band count thresholds in ascending order.
        - :`scores`: Preference scores; `len(scores) == len(thresholds) + 1`.
        - :`id`: Optional unique identifier.
        """
        super().__init__(SpectralPreferenceStrategies.BAND_COUNT.value, id)

        assert wavelength_range is None or (
            isinstance(wavelength_range, (tuple, list)) and len(wavelength_range) == 2
            and wavelength_range[0] < wavelength_range[1]
        ), "wavelength_range must be a (min_nm, max_nm) tuple or None"
        assert isinstance(thresholds, list) and all(isinstance(t, (int, float)) for t in thresholds), \
            "Thresholds must be a list of numbers"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), \
            "Thresholds must be in ascending order"
        assert isinstance(scores, list) and len(scores) == len(thresholds) + 1, \
            "scores must have length len(thresholds) + 1"
        assert all(0.0 <= s <= 1.0 for s in scores), "Scores must be in [0, 1]"

        self.wavelength_range = tuple(wavelength_range) if wavelength_range is not None else None
        self.thresholds = list(thresholds)
        self.scores = list(scores)

    def _eval_preference_function(self, bands: List[Tuple]) -> float:
        self._validate_bands(bands)
        count = len(self._filter_bands(bands, self.wavelength_range))
        for threshold, score in zip(self.thresholds, self.scores[:-1]):
            if count < threshold:
                return score
        return self.scores[-1]

    def __repr__(self):
        return (super().__repr__()[:-1] +
                f", wavelength_range={self.wavelength_range}, thresholds={self.thresholds}, scores={self.scores})")

    def to_dict(self):
        d = super().to_dict()
        d['wavelength_range'] = list(self.wavelength_range) if self.wavelength_range is not None else None
        d['thresholds'] = self.thresholds
        d['scores'] = self.scores
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpectralBandCountRequirement':
        required_keys = ['req_type', 'strategy', 'thresholds', 'scores']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpectralPreferenceStrategies.BAND_COUNT.value, \
            f"Strategy does not match requirement definition. Must be '{SpectralPreferenceStrategies.BAND_COUNT.value}'"
        wl = d.get("wavelength_range", None)
        return cls(tuple(wl) if wl is not None else None,
                   d.get("thresholds"), d.get("scores"), d.get("id", None))

    def __eq__(self, other):
        if not (super().__eq__(other) and isinstance(other, SpectralBandCountRequirement)):
            return False
        wl_match = (self.wavelength_range is None and other.wavelength_range is None) or (
            self.wavelength_range is not None and other.wavelength_range is not None and
            all(abs(a - b) < 1e-6 for a, b in zip(self.wavelength_range, other.wavelength_range))
        )
        return (wl_match
                and self.thresholds == other.thresholds
                and all(abs(a - b) < 1e-6 for a, b in zip(self.scores, other.scores)))


class SpectralResolutionRequirement(SpectralRequirement):
    def __init__(self,
                 wavelength_range: Tuple[float, float],
                 thresholds: List[float],
                 scores: List[float],
                 id=None):
        """
        ### Spectral Resolution Requirement

        Evaluates preference based on the finest (minimum) spectral resolution in nm among
        bands within `wavelength_range`. Lower resolution value = finer = better, so
        `scores` should typically decrease as `thresholds` increase.
        - :`wavelength_range`: `(min_nm, max_nm)` filter, or `None` for all bands.
        - :`thresholds`: Resolution thresholds in nm, ascending order.
        - :`scores`: Preference scores; `len(scores) == len(thresholds) + 1`.
        - :`id`: Optional unique identifier.
        """
        super().__init__(SpectralPreferenceStrategies.RESOLUTION.value, id)

        assert wavelength_range is None or (
            isinstance(wavelength_range, (tuple, list)) and len(wavelength_range) == 2
            and wavelength_range[0] < wavelength_range[1]
        ), "wavelength_range must be a (min_nm, max_nm) tuple or None"
        assert isinstance(thresholds, list) and all(isinstance(t, (int, float)) for t in thresholds), \
            "Thresholds must be a list of numbers"
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)), \
            "Thresholds must be in ascending order"
        assert isinstance(scores, list) and len(scores) == len(thresholds) + 1, \
            "scores must have length len(thresholds) + 1"
        assert all(0.0 <= s <= 1.0 for s in scores), "Scores must be in [0, 1]"

        self.wavelength_range = tuple(wavelength_range) if wavelength_range is not None else None
        self.thresholds = list(thresholds)
        self.scores = list(scores)

    def _eval_preference_function(self, bands: List[Tuple]) -> float:
        self._validate_bands(bands)
        filtered = self._filter_bands(bands, self.wavelength_range)
        if not filtered:
            return 0.0
        best_resolution = min(b[2] for b in filtered)
        # Use <= so that a resolution exactly at the threshold earns the better score.
        # (e.g. 5 nm instrument vs 5 nm threshold → scores[0], not scores[1])
        for threshold, score in zip(self.thresholds, self.scores[:-1]):
            if best_resolution <= threshold:
                return score
        return self.scores[-1]

    def __repr__(self):
        return (super().__repr__()[:-1] +
                f", wavelength_range={self.wavelength_range}, thresholds={self.thresholds}, scores={self.scores})")

    def to_dict(self):
        d = super().to_dict()
        d['wavelength_range'] = list(self.wavelength_range) if self.wavelength_range is not None else None
        d['thresholds'] = self.thresholds
        d['scores'] = self.scores
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpectralResolutionRequirement':
        required_keys = ['req_type', 'strategy', 'thresholds', 'scores']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpectralPreferenceStrategies.RESOLUTION.value, \
            f"Strategy does not match requirement definition. Must be '{SpectralPreferenceStrategies.RESOLUTION.value}'"
        wl = d.get("wavelength_range", None)
        return cls(tuple(wl) if wl is not None else None,
                   d.get("thresholds"), d.get("scores"), d.get("id", None))

    def __eq__(self, other):
        if not (super().__eq__(other) and isinstance(other, SpectralResolutionRequirement)):
            return False
        wl_match = (self.wavelength_range is None and other.wavelength_range is None) or (
            self.wavelength_range is not None and other.wavelength_range is not None and
            all(abs(a - b) < 1e-6 for a, b in zip(self.wavelength_range, other.wavelength_range))
        )
        return (wl_match
                and self.thresholds == other.thresholds
                and all(abs(a - b) < 1e-6 for a, b in zip(self.scores, other.scores)))


class SpectralRangeRequirement(SpectralRequirement):
    def __init__(self,
                 required_min_nm: float,
                 required_max_nm: float,
                 id=None):
        """
        ### Spectral Range Requirement

        Evaluates whether any instrument band overlaps `[required_min_nm, required_max_nm]`
        with enough overlap to be spectrally discernible. For each band, the overlap between
        its coverage `[center +/- bandwidth/2]` and the required window is compared against the
        band's FWHM (resolution). Returns `min(1.0, overlap / FWHM)` for the best-matching
        band — 1.0 when overlap spans at least one full resolution element, partial credit for
        narrower overlaps, 0.0 when no band reaches the required window.
        - :`required_min_nm`: Lower edge of the required spectral window (nm).
        - :`required_max_nm`: Upper edge of the required spectral window (nm).
        - :`id`: Optional unique identifier.
        """
        super().__init__(SpectralPreferenceStrategies.RANGE.value, id)

        assert isinstance(required_min_nm, (int, float)) and required_min_nm > 0, \
            "required_min_nm must be a positive number"
        assert isinstance(required_max_nm, (int, float)) and required_max_nm > required_min_nm, \
            "required_max_nm must be greater than required_min_nm"

        self.required_min_nm = float(required_min_nm)
        self.required_max_nm = float(required_max_nm)

    def _eval_preference_function(self, bands: List[Tuple]) -> float:
        self._validate_bands(bands)
        best = 0.0
        for center_nm, bw_nm, fwhm_nm in bands:
            band_lo = center_nm - bw_nm / 2.0
            band_hi = center_nm + bw_nm / 2.0
            overlap = max(0.0, min(band_hi, self.required_max_nm) - max(band_lo, self.required_min_nm))
            best = max(best, min(1.0, overlap / fwhm_nm))
        return best

    def __repr__(self):
        return (super().__repr__()[:-1] +
                f", required_min_nm={self.required_min_nm}, required_max_nm={self.required_max_nm})")

    def to_dict(self):
        d = super().to_dict()
        d['required_min_nm'] = self.required_min_nm
        d['required_max_nm'] = self.required_max_nm
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpectralRangeRequirement':
        required_keys = ['req_type', 'strategy', 'required_min_nm', 'required_max_nm']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpectralPreferenceStrategies.RANGE.value, \
            f"Strategy does not match requirement definition. Must be '{SpectralPreferenceStrategies.RANGE.value}'"
        return cls(d.get("required_min_nm"), d.get("required_max_nm"), d.get("id", None))

    def __eq__(self, other):
        if not (super().__eq__(other) and isinstance(other, SpectralRangeRequirement)):
            return False
        return (abs(self.required_min_nm - other.required_min_nm) < 1e-6 and
                abs(self.required_max_nm - other.required_max_nm) < 1e-6)


class TieredSpectralRequirement(SpectralRequirement):
    def __init__(self, tiers: List[Dict], id=None):
        """
        ### Tiered Spectral Requirement

        Evaluates an ordered list of tiers from best (highest score) to worst. For each tier
        the preference value is `tier_score x product(sub_req.pref for sub_req in requirements)`.
        The returned value is taken from the **first** tier where that product is > 0; if no
        tier passes, 0.0 is returned. Use this to express "ideally X, or at least Y" compound
        requirements where sub-requirements within each tier must be jointly satisfied.

        Example — "ideally ≤5 nm resolution AND full VNIR range; at least full VNIR range":
        ```python
        TieredSpectralRequirement(tiers=[
            {"score": 1.0, "requirements": [SpectralResolutionRequirement((380, 1000), [5], [1.0, 0.0]),
                                            SpectralRangeRequirement(380.0, 2500.0)]},
            {"score": 0.5, "requirements": [SpectralRangeRequirement(380.0, 2500.0)]},
        ])
        ```
        - :`tiers`: List of dicts in descending score order, each containing:
            - `"score"` (float ∈ [0, 1]): multiplied by the product of sub-requirement preferences.
            - `"requirements"` (List[SpectralRequirement]): all preferences are multiplied together.
        - :`id`: Optional unique identifier.
        """
        super().__init__(SpectralPreferenceStrategies.TIERED.value, id)

        assert isinstance(tiers, list) and len(tiers) > 0, "Tiers must be a non-empty list"
        for tier in tiers:
            assert isinstance(tier, dict) and "score" in tier and "requirements" in tier, \
                "Each tier must be a dict with 'score' and 'requirements' keys"
            assert isinstance(tier["score"], (int, float)) and 0.0 <= tier["score"] <= 1.0, \
                "Tier score must be in [0, 1]"
            assert isinstance(tier["requirements"], list) and len(tier["requirements"]) > 0, \
                "Each tier must have a non-empty list of requirements"
            assert all(isinstance(r, SpectralRequirement) for r in tier["requirements"]), \
                "All tier requirements must be SpectralRequirement instances"
        scores = [t["score"] for t in tiers]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
            "Tier scores must be in descending order (best first)"

        self.tiers = tiers

    def _eval_preference_function(self, bands: List[Tuple]) -> float:
        for tier in self.tiers:
            product = 1.0
            for req in tier["requirements"]:
                product *= req._eval_preference_function(bands)
            value = tier["score"] * product
            if value > 0.0:
                return value
        return 0.0

    def __repr__(self):
        summaries = ', '.join(f"(score={t['score']}, n_reqs={len(t['requirements'])})" for t in self.tiers)
        return super().__repr__()[:-1] + f", tiers=[{summaries}])"

    def to_dict(self):
        d = super().to_dict()
        d['tiers'] = [
            {"score": t["score"], "requirements": [r.to_dict() for r in t["requirements"]]}
            for t in self.tiers
        ]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'TieredSpectralRequirement':
        required_keys = ['req_type', 'strategy', 'tiers']
        assert all(key in d for key in required_keys), \
            f"Dictionary must contain the keys: {required_keys}"
        assert d.get("strategy").lower() == SpectralPreferenceStrategies.TIERED.value, \
            f"Strategy does not match requirement definition. Must be '{SpectralPreferenceStrategies.TIERED.value}'"
        tiers = [
            {"score": t["score"],
             "requirements": [SpectralRequirement.from_dict(r) for r in t["requirements"]]}
            for t in d.get("tiers")
        ]
        return cls(tiers, d.get("id", None))

    def __eq__(self, other):
        if not (super().__eq__(other) and isinstance(other, TieredSpectralRequirement)):
            return False
        if len(self.tiers) != len(other.tiers):
            return False
        for t1, t2 in zip(self.tiers, other.tiers):
            if abs(t1["score"] - t2["score"]) > 1e-6:
                return False
            if len(t1["requirements"]) != len(t2["requirements"]):
                return False
            if not all(r1 == r2 for r1, r2 in zip(t1["requirements"], t2["requirements"])):
                return False
        return True