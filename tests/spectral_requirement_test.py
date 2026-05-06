import unittest

from execsatm.requirements import (
    MissionRequirement,
    RequirementTypes,
    SpectralPreferenceStrategies,
    SpectralRequirement,
    SpectralBandCountRequirement,
    SpectralResolutionRequirement,
    SpectralRangeRequirement,
    TieredSpectralRequirement,
)
from execsatm.utils import print_banner

ATTRIBUTE = SpectralRequirement.ATTRIBUTE  # "spectral_bands"

"""
----------------------------------
SPECTRAL BAND COUNT REQUIREMENT
----------------------------------
"""
class TestSpectralBandCountRequirement(unittest.TestCase):
    def setUp(self):
        self.wavelength_range = (380, 1000)
        self.thresholds = [3, 5]
        self.scores = [0.0, 0.5, 1.0]
        self.req = SpectralBandCountRequirement(
            wavelength_range=self.wavelength_range,
            thresholds=self.thresholds,
            scores=self.scores,
        )

    def test_constructor(self):
        self.assertIsInstance(self.req, SpectralBandCountRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPECTRAL.value)
        self.assertEqual(self.req.attribute, ATTRIBUTE)
        self.assertEqual(self.req.strategy, SpectralPreferenceStrategies.BAND_COUNT.value)
        self.assertEqual(self.req.wavelength_range, self.wavelength_range)
        self.assertEqual(self.req.thresholds, self.thresholds)
        self.assertEqual(self.req.scores, self.scores)

        # wavelength_range=None is valid (count all bands)
        req_no_range = SpectralBandCountRequirement(
            wavelength_range=None, thresholds=[2], scores=[0.0, 1.0]
        )
        self.assertIsNone(req_no_range.wavelength_range)

        # invalid wavelength_range
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range="not_a_tuple",
                          thresholds=self.thresholds, scores=self.scores)
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=(1000, 380),     # min >= max
                          thresholds=self.thresholds, scores=self.scores)

        # invalid thresholds
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds="not_a_list", scores=self.scores)
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=[5, 3],               # not ascending
                          scores=self.scores)
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=[3, 5, 7],            # length mismatch with scores
                          scores=self.scores)

        # invalid scores
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores="not_a_list")
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=[0.0, 1.5, 1.0])  # out of [0,1]
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=[0.0, 1.0])       # wrong length

        # invalid id
        self.assertRaises(AssertionError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=self.scores, id=123)
        self.assertRaises(ValueError, SpectralBandCountRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=self.scores, id="123")

    def test_get_preference(self):
        # thresholds=[3,5], scores=[0.0, 0.5, 1.0] over range (380,1000)
        no_bands      = []
        one_band      = [(500, 100, 10)]
        three_bands   = [(450, 50, 5), (650, 50, 5), (850, 50, 5)]
        five_bands    = [(450,50,5),(550,50,5),(650,50,5),(750,50,5),(850,50,5)]
        out_of_range  = [(1500, 100, 10), (2000, 100, 10)]  # all outside (380,1000)

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, no_bands),     0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, one_band),     0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, three_bands),  0.5)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, five_bands),   1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, out_of_range), 0.0)

        # mixed: 3 in range, 2 out of range → 0.5
        mixed = three_bands + out_of_range
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, mixed), 0.5)

        # wavelength_range=None counts all bands
        req_all = SpectralBandCountRequirement(None, [3], [0.0, 1.0])
        self.assertAlmostEqual(req_all.calc_preference(ATTRIBUTE, out_of_range), 0.0)  # only 2
        self.assertAlmostEqual(req_all.calc_preference(ATTRIBUTE, three_bands),  1.0)

        # wrong attribute type/name
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, one_band)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", one_band)

        # invalid band list
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, "not_a_list")
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(500, 100)])      # band missing resolution
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(-500, 100, 5)])  # negative center

    def test_representation(self):
        expected = (
            "SpectralRequirement(strategy=BAND_COUNT, "
            f"wavelength_range={self.wavelength_range}, "
            f"thresholds={self.thresholds}, scores={self.scores})"
        )
        self.assertEqual(repr(self.req), expected)

    def test_to_dict(self):
        d = self.req.to_dict()
        self.assertEqual(d["req_type"], RequirementTypes.SPECTRAL.value)
        self.assertEqual(d["attribute"], ATTRIBUTE)
        self.assertEqual(d["strategy"], SpectralPreferenceStrategies.BAND_COUNT.value)
        self.assertEqual(d["wavelength_range"], list(self.wavelength_range))
        self.assertEqual(d["thresholds"], self.thresholds)
        self.assertEqual(d["scores"], self.scores)
        self.assertEqual(d["id"], self.req.id)

    def test_from_dict(self):
        req_dict = {
            "req_type": RequirementTypes.SPECTRAL.value,
            "attribute": ATTRIBUTE,
            "strategy": SpectralPreferenceStrategies.BAND_COUNT.value,
            "wavelength_range": list(self.wavelength_range),
            "thresholds": self.thresholds,
            "scores": self.scores,
        }
        # class method
        r = SpectralBandCountRequirement.from_dict(req_dict)
        self.assertIsInstance(r, SpectralBandCountRequirement)
        self.assertEqual(r.wavelength_range, self.req.wavelength_range)
        self.assertEqual(r.thresholds, self.req.thresholds)
        self.assertEqual(r.scores, self.req.scores)
        self.assertNotEqual(r.id, self.req.id)

        # parent class method
        r2 = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(r2, SpectralBandCountRequirement)
        self.assertEqual(r2.wavelength_range, self.req.wavelength_range)
        self.assertEqual(r2.thresholds, self.req.thresholds)
        self.assertEqual(r2.scores, self.req.scores)
        self.assertNotEqual(r2.id, self.req.id)

        # wavelength_range=None round-trips correctly
        req_none = SpectralBandCountRequirement(None, [2], [0.0, 1.0])
        self.assertIsNone(MissionRequirement.from_dict(req_none.to_dict()).wavelength_range)

    def test_copy(self):
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, SpectralBandCountRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)


"""
----------------------------------
SPECTRAL RESOLUTION REQUIREMENT
----------------------------------
"""
class TestSpectralResolutionRequirement(unittest.TestCase):
    def setUp(self):
        self.wavelength_range = (380, 1000)
        self.thresholds = [5, 10]
        self.scores = [1.0, 0.5, 0.0]   # lower resolution (nm) = better
        self.req = SpectralResolutionRequirement(
            wavelength_range=self.wavelength_range,
            thresholds=self.thresholds,
            scores=self.scores,
        )

    def test_constructor(self):
        self.assertIsInstance(self.req, SpectralResolutionRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPECTRAL.value)
        self.assertEqual(self.req.attribute, ATTRIBUTE)
        self.assertEqual(self.req.strategy, SpectralPreferenceStrategies.RESOLUTION.value)
        self.assertEqual(self.req.wavelength_range, self.wavelength_range)
        self.assertEqual(self.req.thresholds, self.thresholds)
        self.assertEqual(self.req.scores, self.scores)

        # wavelength_range=None is valid
        SpectralResolutionRequirement(None, [10], [1.0, 0.0])

        # invalid wavelength_range
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=(1000, 380),
                          thresholds=self.thresholds, scores=self.scores)
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range="not_a_tuple",
                          thresholds=self.thresholds, scores=self.scores)

        # invalid thresholds
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds="not_a_list", scores=self.scores)
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=[10, 5],              # not ascending
                          scores=self.scores)
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=[5, 10, 20],          # length mismatch
                          scores=self.scores)

        # invalid scores
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores="not_a_list")
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=[1.0, 1.5, 0.0])  # out of [0,1]
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=[1.0, 0.0])       # wrong length

        # invalid id
        self.assertRaises(AssertionError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=self.scores, id=123)
        self.assertRaises(ValueError, SpectralResolutionRequirement,
                          wavelength_range=self.wavelength_range,
                          thresholds=self.thresholds, scores=self.scores, id="123")

    def test_get_preference(self):
        # thresholds=[5, 10], scores=[1.0, 0.5, 0.0]; uses <=
        # best_res <= 5  → 1.0
        # best_res <= 10 → 0.5
        # best_res > 10  → 0.0
        bands_5nm  = [(500, 100, 5),  (700, 100, 5)]    # best = 5nm  → 1.0
        bands_8nm  = [(500, 100, 8),  (700, 100, 8)]    # best = 8nm  → 0.5
        bands_15nm = [(500, 100, 15)]                   # best = 15nm → 0.0
        no_in_range = [(1500, 100, 5)]                  # outside (380,1000) → 0.0

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_5nm),   1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_8nm),   0.5)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_15nm),  0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, no_in_range), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, []),          0.0)

        # takes the best (min) resolution among filtered bands
        mixed = [(500, 100, 5), (700, 100, 12)]   # best in range = 5nm → 1.0
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, mixed), 1.0)

        # wavelength_range=None uses all bands
        req_all = SpectralResolutionRequirement(None, [10], [1.0, 0.0])
        self.assertAlmostEqual(req_all.calc_preference(ATTRIBUTE, no_in_range), 1.0)  # 5nm resolution

        # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, bands_5nm)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", bands_5nm)

        # invalid band list
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, "not_a_list")
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(500, 100)])
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(0, 100, 5)])   # center=0

    def test_representation(self):
        expected = (
            "SpectralRequirement(strategy=RESOLUTION, "
            f"wavelength_range={self.wavelength_range}, "
            f"thresholds={self.thresholds}, scores={self.scores})"
        )
        self.assertEqual(repr(self.req), expected)

    def test_to_dict(self):
        d = self.req.to_dict()
        self.assertEqual(d["req_type"], RequirementTypes.SPECTRAL.value)
        self.assertEqual(d["attribute"], ATTRIBUTE)
        self.assertEqual(d["strategy"], SpectralPreferenceStrategies.RESOLUTION.value)
        self.assertEqual(d["wavelength_range"], list(self.wavelength_range))
        self.assertEqual(d["thresholds"], self.thresholds)
        self.assertEqual(d["scores"], self.scores)
        self.assertEqual(d["id"], self.req.id)

    def test_from_dict(self):
        req_dict = {
            "req_type": RequirementTypes.SPECTRAL.value,
            "attribute": ATTRIBUTE,
            "strategy": SpectralPreferenceStrategies.RESOLUTION.value,
            "wavelength_range": list(self.wavelength_range),
            "thresholds": self.thresholds,
            "scores": self.scores,
        }
        # class method
        r = SpectralResolutionRequirement.from_dict(req_dict)
        self.assertIsInstance(r, SpectralResolutionRequirement)
        self.assertEqual(r.wavelength_range, self.req.wavelength_range)
        self.assertEqual(r.thresholds, self.req.thresholds)
        self.assertEqual(r.scores, self.req.scores)
        self.assertNotEqual(r.id, self.req.id)

        # parent class method
        r2 = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(r2, SpectralResolutionRequirement)
        self.assertEqual(r2.wavelength_range, self.req.wavelength_range)
        self.assertEqual(r2.thresholds, self.req.thresholds)
        self.assertEqual(r2.scores, self.req.scores)
        self.assertNotEqual(r2.id, self.req.id)

    def test_copy(self):
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, SpectralResolutionRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)


"""
----------------------------------
SPECTRAL RANGE REQUIREMENT
----------------------------------
"""
class TestSpectralRangeRequirement(unittest.TestCase):
    def setUp(self):
        self.required_min_nm = 380.0
        self.required_max_nm = 2500.0
        self.req = SpectralRangeRequirement(
            required_min_nm=self.required_min_nm,
            required_max_nm=self.required_max_nm,
        )

    def test_constructor(self):
        self.assertIsInstance(self.req, SpectralRangeRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPECTRAL.value)
        self.assertEqual(self.req.attribute, ATTRIBUTE)
        self.assertEqual(self.req.strategy, SpectralPreferenceStrategies.RANGE.value)
        self.assertAlmostEqual(self.req.required_min_nm, self.required_min_nm)
        self.assertAlmostEqual(self.req.required_max_nm, self.required_max_nm)

        # invalid required_min_nm
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm="not_a_float", required_max_nm=self.required_max_nm)
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm=0,             required_max_nm=self.required_max_nm)  # must be > 0
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm=-100,          required_max_nm=self.required_max_nm)

        # invalid required_max_nm
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm=self.required_min_nm, required_max_nm="not_a_float")
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm=self.required_min_nm, required_max_nm=100)  # max <= min

        # invalid id
        self.assertRaises(AssertionError, SpectralRangeRequirement,
                          required_min_nm=self.required_min_nm,
                          required_max_nm=self.required_max_nm, id=123)
        self.assertRaises(ValueError, SpectralRangeRequirement,
                          required_min_nm=self.required_min_nm,
                          required_max_nm=self.required_max_nm, id="123")

    def test_get_preference(self):
        # edges = center ± bandwidth/2
        # wide: min=375, max=2550 → covers [380, 2500] → 1.0
        wide   = [(500, 250, 10), (1500, 500, 20), (2400, 300, 30)]
        # exact: min=375 (380-10/2), max=2505 (2500+10/2) → 1.0
        exact  = [(380, 10, 5), (2500, 10, 5)]
        # narrow: max=850 < 2500 → 0.0
        narrow = [(500, 100, 10), (800, 100, 10)]
        # no low end: min=450 > 380 → 0.0
        no_low = [(600, 100, 10), (2500, 10, 5)]
        # empty
        empty  = []

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, wide),   1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, exact),  1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, narrow), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, no_low), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, empty),  0.0)

        # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, wide)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", wide)

        # invalid band list
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, "not_a_list")
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(500, 100)])
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(0, 100, 5)])

    def test_representation(self):
        expected = (
            f"SpectralRequirement(strategy=RANGE, "
            f"required_min_nm={self.required_min_nm}, "
            f"required_max_nm={self.required_max_nm})"
        )
        self.assertEqual(repr(self.req), expected)

    def test_to_dict(self):
        d = self.req.to_dict()
        self.assertEqual(d["req_type"], RequirementTypes.SPECTRAL.value)
        self.assertEqual(d["attribute"], ATTRIBUTE)
        self.assertEqual(d["strategy"], SpectralPreferenceStrategies.RANGE.value)
        self.assertAlmostEqual(d["required_min_nm"], self.required_min_nm)
        self.assertAlmostEqual(d["required_max_nm"], self.required_max_nm)
        self.assertEqual(d["id"], self.req.id)

    def test_from_dict(self):
        req_dict = {
            "req_type": RequirementTypes.SPECTRAL.value,
            "attribute": ATTRIBUTE,
            "strategy": SpectralPreferenceStrategies.RANGE.value,
            "required_min_nm": self.required_min_nm,
            "required_max_nm": self.required_max_nm,
        }
        # class method
        r = SpectralRangeRequirement.from_dict(req_dict)
        self.assertIsInstance(r, SpectralRangeRequirement)
        self.assertAlmostEqual(r.required_min_nm, self.req.required_min_nm)
        self.assertAlmostEqual(r.required_max_nm, self.req.required_max_nm)
        self.assertNotEqual(r.id, self.req.id)

        # parent class method
        r2 = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(r2, SpectralRangeRequirement)
        self.assertAlmostEqual(r2.required_min_nm, self.req.required_min_nm)
        self.assertAlmostEqual(r2.required_max_nm, self.req.required_max_nm)
        self.assertNotEqual(r2.id, self.req.id)

    def test_copy(self):
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, SpectralRangeRequirement)
        self.assertEqual(req_copy.to_dict(), self.req.to_dict())
        self.assertIsNot(req_copy, self.req)


"""
----------------------------------
TIERED SPECTRAL REQUIREMENT
----------------------------------
"""
class TestTieredSpectralRequirement(unittest.TestCase):
    def setUp(self):
        self.tier1_req = SpectralBandCountRequirement((8000, 12000), [5], [0.0, 1.0])
        self.tier2_req = SpectralBandCountRequirement((8000, 12000), [3], [0.0, 1.0])
        self.tiers = [
            {"score": 1.0, "requirements": [self.tier1_req]},
            {"score": 0.5, "requirements": [self.tier2_req]},
        ]
        self.req = TieredSpectralRequirement(tiers=self.tiers)

    def test_constructor(self):
        self.assertIsInstance(self.req, TieredSpectralRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPECTRAL.value)
        self.assertEqual(self.req.attribute, ATTRIBUTE)
        self.assertEqual(self.req.strategy, SpectralPreferenceStrategies.TIERED.value)
        self.assertEqual(len(self.req.tiers), 2)

        # multiple requirements in a single tier is valid
        multi_req = TieredSpectralRequirement(tiers=[{
            "score": 1.0,
            "requirements": [self.tier1_req, self.tier2_req],
        }])
        self.assertEqual(len(multi_req.tiers[0]["requirements"]), 2)

        # invalid tiers: empty list
        self.assertRaises(AssertionError, TieredSpectralRequirement, tiers=[])

        # invalid tier: not a dict
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=["not_a_dict"])

        # invalid tier: missing "score"
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"requirements": [self.tier1_req]}])

        # invalid tier: missing "requirements"
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.0}])

        # invalid tier: score out of [0, 1]
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.5, "requirements": [self.tier1_req]}])

        # invalid tier: requirements not SpectralRequirement instances
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.0, "requirements": ["not_a_req"]}])

        # invalid tier: scores not in descending order
        self.assertRaises(AssertionError, TieredSpectralRequirement, tiers=[
            {"score": 0.5, "requirements": [self.tier2_req]},
            {"score": 1.0, "requirements": [self.tier1_req]},  # ascending → invalid
        ])

        # invalid id
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=self.tiers, id=123)
        self.assertRaises(ValueError, TieredSpectralRequirement,
                          tiers=self.tiers, id="123")

    def test_get_preference(self):
        # tier1: ≥5 bands in 8-12 um → 1.0; tier2: ≥3 bands → 0.5; else → 0.0
        five_bands  = [(8500,500,100),(9500,500,100),(10500,500,100),(11500,500,100),(12000,500,100)]
        three_bands = [(8500,500,100),(10000,500,100),(11500,500,100)]
        one_band    = [(9000,500,100)]
        vnir_bands  = [(500,100,10),(700,100,10)]  # outside 8-12 um → 0.0

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, five_bands),  1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, three_bands), 0.5)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, one_band),    0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, vnir_bands),  0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, []),          0.0)

        # tier with multiple simultaneous requirements: both must pass
        tir_req  = SpectralBandCountRequirement((8000, 12000), [3], [0.0, 1.0])
        mwir_req = SpectralBandCountRequirement((3000, 4500),  [1], [0.0, 1.0])
        multi_tier = TieredSpectralRequirement(tiers=[
            {"score": 1.0, "requirements": [tir_req, mwir_req]},
        ])
        tir_only  = three_bands                                      # no MWIR
        both_reqs = three_bands + [(3500, 200, 100)]                 # TIR + MWIR
        self.assertAlmostEqual(multi_tier.calc_preference(ATTRIBUTE, tir_only),  0.0)
        self.assertAlmostEqual(multi_tier.calc_preference(ATTRIBUTE, both_reqs), 1.0)

        # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, five_bands)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", five_bands)

        # invalid band list
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, "not_a_list")
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(500, 100)])

    def test_representation(self):
        expected = (
            "SpectralRequirement(strategy=TIERED, "
            "tiers=[(score=1.0, n_reqs=1), (score=0.5, n_reqs=1)])"
        )
        self.assertEqual(repr(self.req), expected)

    def test_to_dict(self):
        d = self.req.to_dict()
        self.assertEqual(d["req_type"], RequirementTypes.SPECTRAL.value)
        self.assertEqual(d["attribute"], ATTRIBUTE)
        self.assertEqual(d["strategy"], SpectralPreferenceStrategies.TIERED.value)
        self.assertEqual(d["id"], self.req.id)
        self.assertEqual(len(d["tiers"]), 2)
        self.assertAlmostEqual(d["tiers"][0]["score"], 1.0)
        self.assertAlmostEqual(d["tiers"][1]["score"], 0.5)
        self.assertEqual(len(d["tiers"][0]["requirements"]), 1)
        self.assertEqual(len(d["tiers"][1]["requirements"]), 1)

    def test_from_dict(self):
        req_dict = {
            "req_type": RequirementTypes.SPECTRAL.value,
            "attribute": ATTRIBUTE,
            "strategy": SpectralPreferenceStrategies.TIERED.value,
            "tiers": [
                {
                    "score": 1.0,
                    "requirements": [{
                        "req_type": RequirementTypes.SPECTRAL.value,
                        "attribute": ATTRIBUTE,
                        "strategy": SpectralPreferenceStrategies.BAND_COUNT.value,
                        "wavelength_range": [8000, 12000],
                        "thresholds": [5],
                        "scores": [0.0, 1.0],
                    }],
                },
                {
                    "score": 0.5,
                    "requirements": [{
                        "req_type": RequirementTypes.SPECTRAL.value,
                        "attribute": ATTRIBUTE,
                        "strategy": SpectralPreferenceStrategies.BAND_COUNT.value,
                        "wavelength_range": [8000, 12000],
                        "thresholds": [3],
                        "scores": [0.0, 1.0],
                    }],
                },
            ],
        }
        # class method
        r = TieredSpectralRequirement.from_dict(req_dict)
        self.assertIsInstance(r, TieredSpectralRequirement)
        self.assertEqual(len(r.tiers), 2)
        self.assertAlmostEqual(r.tiers[0]["score"], 1.0)
        self.assertAlmostEqual(r.tiers[1]["score"], 0.5)
        self.assertIsInstance(r.tiers[0]["requirements"][0], SpectralBandCountRequirement)
        self.assertNotEqual(r.id, self.req.id)

        # parent class method
        r2 = MissionRequirement.from_dict(req_dict)
        self.assertIsInstance(r2, TieredSpectralRequirement)
        self.assertEqual(len(r2.tiers), 2)
        self.assertNotEqual(r2.id, self.req.id)

    def test_copy(self):
        req_copy = self.req.copy()
        self.assertIsInstance(req_copy, TieredSpectralRequirement)
        self.assertEqual(req_copy, self.req)
        self.assertIsNot(req_copy, self.req)


if __name__ == "__main__":
    print_banner("Spectral Requirement Definition Test")
    unittest.main()
