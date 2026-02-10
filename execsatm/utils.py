import os
import numpy as np 

class Interval:
    """ Represents an linear interval set of real numbers """

    def __init__(self, left:float, right:float, left_open:bool=False, right_open:bool=False):
        # validate inputs
        if not isinstance(left, (float, int)) and not np.isscalar(left):
            raise AttributeError(f'`left` must be of type `float` or type `int`. is of type {type(left)}.')
        if not isinstance(right, (float, int)) and not np.isscalar(right):
            raise AttributeError(f'`right` must be of type `float` or type `int`. is of type {type(right)}.')
        if not isinstance(left_open, bool):
            raise AttributeError(f'`left_open` must be of type `bool`. is of type {type(left_open)}.')
        if not isinstance(right_open, bool):
            raise AttributeError(f'`right_open` must be of type `bool`. is of type {type(right_open)}.')

        # assign attributes
        self.left : float = left
        self.left_open : bool = left_open if not np.isneginf(left) else True

        self.right : float = right
        self.right_open : bool = right_open if not np.isinf(right) else True

        if self.right < self.left:
            raise ValueError('The right side of interval must be later than the left side of the interval.')

    def __contains__(self, x: float) -> bool:
        """ checks if `x` is contained in the interval """
        l = self.left < x if self.left_open else self.left < x or abs(self.left - x) < 1e-6
        r = x < self.right if self.right_open else x < self.right or abs(self.right - x) < 1e-6        
        return l and r

    def is_after(self, x : float) -> bool:
        """ checks if the interval starts after the value `x` """
        return x < self.left if self.left_open else (x < self.left or abs(self.left - x) < 1e-6)

    def is_before(self, x : float) -> bool:
        """ checks if the interval ends before the value `x` """
        return self.right < x if self.right_open else (self.right < x or abs(self.right - x) < 1e-6)

    def is_empty(self) -> bool:
        """ checks if the interval is empty """
        return (abs(self.left - self.right) < 1e-6 and (self.left_open or self.right_open)) or (self.left > self.right)

    def overlaps(self, __other : 'Interval') -> bool:
        """ checks if this interval has an overlap with another """
        
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot check overlap with object of type `{type(__other)}`.')

        return ( self.left in __other
                or self.right in __other
                or __other.left in self
                or __other.right in self)

    def is_subset(self, _other: 'Interval') -> bool:
        """ checks if this interval is a subset of another """
        if not isinstance(_other, Interval):
            raise TypeError(f'Cannot check subset with object of type `{type(_other)}`.')

        if _other.left_open and self.left <= _other.left:                
            return False
        if not _other.left_open and self.left < _other.left:
            return False
            
        if _other.right_open and _other.right <= self.right:
            return False
        if not _other.right_open and _other.right < self.right:
            return False
            
        return True

        # return (__other.left <= self.left and self.right <= __other.right
        #         and (self.left_open or not __other.left_open)
        #         and (self.right_open or not __other.right_open))

    def intersection(self, __other : 'Interval') -> 'Interval':
        """ returns the intersect of two intervals """

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot merge with object of type `{type(__other)}`.')

        if not self.overlaps(__other): return EmptyInterval()

        # find the left and right bounds of the intersection
        left = max(self.left, __other.left)
        right = min(self.right, __other.right)

        # check if the left and right bounds are open
        left_open = self.left_open if left == self.left else __other.left_open
        right_open = __other.right_open if right == __other.right else self.right_open

        # compensate for rounding errors 
        if left > right and abs(left - right) <= 1e-6:
            l = min(left, right)
            r = max(left, right)
            left = l
            right = r

        # create a new interval object
        return Interval(left, right, left_open, right_open)

    def union(self, __other : 'Interval', extend : bool = False) -> 'Interval':
        """ returns the union of this and another interval """

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot merge with object of type `{type(__other)}`.')

        # if the intervals do not overlap, return an empty interval
        if not self.overlaps(__other) and not extend: return EmptyInterval()

        # find the left and right bounds of the union
        left = min(self.left, __other.left)
        right = max(self.right, __other.right)

        # check if the left and right bounds are open
        left_open = self.left_open if left == self.left else __other.left_open
        right_open = __other.right_open if right == __other.right else self.right_open

        # create a new interval object
        return Interval(left, right, left_open, right_open)
    
    def join(self, __other : 'Interval') -> None:
        """ joins another interval into this interval """

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot merge with object of type `{type(__other)}`.')

        # if the intervals do not overlap, return an empty interval
        if not self.overlaps(__other) and not extend: return EmptyInterval()

        # find the left and right bounds of the union
        left = min(self.left, __other.left)
        right = max(self.right, __other.right)

        # check if the left and right bounds are open
        left_open = self.left_open if left == self.left else __other.left_open
        right_open = __other.right_open if right == __other.right else self.right_open

        # update this interval's bounds
        self.left = left
        self.right = right
        self.left_open = left_open
        self.right_open = right_open

        # return nothing
        return

    def extend(self, x: float, open:bool=False) -> None:
        """ extends bounds of interval to include new value `x` """

        if x < self.left:
            # extend left bound
            self.left = x
            self.left_open = open
        
        elif x > self.right:
            # extend right bound
            self.right = x
            self.right_open = open

        return
    
    def update_left(self, left: float, open: bool = False) -> None:
        """ updates the left bound of the interval """
        if left <= self.right:
            self.left = left
            self.left_open = open
        else:
            raise ValueError('Cannot update left bound to a value greater than the right bound.')
        
    def open_left(self) -> None:
        """ opens the left bound of the interval """
        self.left_open = True

    def close_left(self) -> None:
        """ closes the left bound of the interval """
        self.left_open = False
        
    def update_right(self, right: float, open: bool = False) -> None:
        """ updates the right bound of the interval """
        if right >= self.left:
            self.right = right
            self.right_open = open
        else:
            raise ValueError('Cannot update right bound to a value less than the left bound.')
        
    def open_right(self) -> None:
        """ opens the right bound of the interval """
        self.right_open = True

    def close_right(self) -> None:
        """ closes the right bound of the interval """
        self.right_open = False
    
    def span(self) -> float:
        """ returns the span of the interval """       
        return self.right - self.left

    def __eq__(self, __other: 'Interval') -> bool:

        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')

        return (abs(self.left - __other.left) < 1e-6 
                and abs(self.right - __other.right) < 1e-6 
                and self.left_open == __other.left_open 
                and self.right_open == __other.right_open)

    def __gt__(self, __other: 'Interval') -> bool:
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')

        if abs(self.left - __other.left) < 1e-6:
            return self.span() > __other.span()
        
        return self.left > __other.left

    def __ge__(self, __other: 'Interval') -> bool:
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')

        if abs(self.left - __other.left) < 1e-6:
            return self.span() >= __other.span()
        
        return self.left >= __other.left
    
    def __lt__(self, __other: 'Interval') -> bool:
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')
        
        if abs(self.left - __other.left) < 1e-6:
            return self.span() < __other.span()
        
        return self.left < __other.left

    def __le__(self, __other: 'Interval') -> bool:
        if not isinstance(__other, Interval):
            raise TypeError(f'Cannot compare with object of type `{type(__other)}`.')
        
        if abs(self.left - __other.left) < 1e-6:
            return self.span() <= __other.span()
        
        return self.left <= __other.left
    
    def __repr__(self) -> str:
        l_bracket = '(' if self.left_open else '['
        r_bracket = ')' if self.right_open else ']'
        return f'Interval{l_bracket}{self.left},{self.right}{r_bracket}'
    
    def __str__(self) -> str:
        l_bracket = '(' if self.left_open else '['
        r_bracket = ')' if self.right_open else ']'
        return f'{l_bracket}{np.round(self.left,3)},{np.round(self.right,3)}{r_bracket}'
    
    def __hash__(self) -> int:
        return hash(repr(self))
    
    def copy(self) -> object:
        """ Create a deep copy of the interval. """
        return Interval(self.left, self.right, self.left_open, self.right_open)

    def to_dict(self) -> dict:
        """ Convert the interval to a dictionary. """
        return {
            "left": self.left,
            "right": self.right,
            "left_open": self.left_open,
            "right_open": self.right_open
        }

    @classmethod
    def from_dict(cls, interval_dict: dict) -> 'Interval':
        """ Create an interval from a dictionary. """
        return cls(interval_dict['left'], interval_dict['right'], interval_dict['left_open'], interval_dict['right_open'])

class EmptyInterval(Interval):
    """ Represents an empty interval """

    def __init__(self):
        super().__init__(np.NAN, np.NAN, True, True)

    def is_empty(self) -> bool:
        return True

    def __repr__(self) -> str:
        return 'EmptyInterval()'

    def __contains__(self, x: float) -> bool:
        return False

    def span(self) -> float:
        """ Returns the span of the interval. """
        return 0.0
    
def print_banner(scenario_name = None) -> None:
    # clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # construct banner string
    out = "\n======================================================"
    # out += '\n  ______                _____      _______ __  __ \n|  ____|              / ____|  /\|__   __|  \/  |\n| |__  __  _____  ___| (___   /  \  | |  | \  / |\n|  __| \ \/ / _ \/ __|\___ \ / /\ \ | |  | |\/| |\n| |____ >  <  __/ (__ ____) / ____ \| |  | |  | |\n|______/_/\_\___|\___|_____/_/    \_\_|  |_|  |_| (v1.0)'
    out += '''\n _____              _____  ___ ________  ___
|  ___|            /  ___|/ _ \_   _|  \/  |
| |____  _____  ___\ `--./ /_\ \| | | .  . |
|  __\ \/ / _ \/ __|`--. \  _  || | | |\/| |
| |___>  <  __/ (__/\__/ / | | || | | |  | |
\____/_/\_\___|\___\____/\_| |_/\_/ \_|  |_/ (v1.0)
'''
    out += "\n======================================================"
    out += '\n\tTexas A&M University - SEAK Lab ©'
    out += "\n======================================================"

    # include scenario name if provided
    if scenario_name is not None: out += f"\nSCENARIO: {scenario_name}"
    
    # print the banner
    print(out)