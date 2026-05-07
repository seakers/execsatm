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

# Band tuple format: (center_nm, bandwidth_nm, resolution_nm) — all values in nm.
# Edge wavelengths are derived as: lower = center - bandwidth/2, upper = center + bandwidth/2.
# wavelength_range tuples are (min_nm, max_nm).

"""
----------------------------------
SPECTRAL BAND COUNT REQUIREMENT
----------------------------------
"""
class TestSpectralBandCountRequirement(unittest.TestCase):
    def setUp(self):
        self.wavelength_range = (380, 1000)  # nm
        self.thresholds = [3, 5]            # band counts
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
        # thresholds=[3,5] bands, scores=[0.0, 0.5, 1.0] over range (380, 1000) nm
        no_bands      = []
        one_band      = [(500, 100, 10)]                                            # center=500nm, bw=100nm, res=10nm
        three_bands   = [(450, 50, 5), (650, 50, 5), (850, 50, 5)]
        five_bands    = [(450,50,5),(550,50,5),(650,50,5),(750,50,5),(850,50,5)]
        out_of_range  = [(1500, 100, 10), (2000, 100, 10)]  # centers outside (380, 1000) nm

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
        self.wavelength_range = (380, 1000)  # nm
        self.thresholds = [5, 10]            # nm — resolution thresholds (lower = finer)
        self.scores = [1.0, 0.5, 0.0]        # descending: finer resolution earns higher score
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
        # thresholds=[5, 10] nm, scores=[1.0, 0.5, 0.0]; uses <=
        # best_res <= 5nm  → 1.0
        # best_res <= 10nm → 0.5
        # best_res > 10nm  → 0.0
        bands_5nm   = [(500, 100, 5),  (700, 100, 5)]   # best resolution = 5nm  → 1.0
        bands_8nm   = [(500, 100, 8),  (700, 100, 8)]   # best resolution = 8nm  → 0.5
        bands_15nm  = [(500, 100, 15)]                  # best resolution = 15nm → 0.0
        no_in_range = [(1500, 100, 5)]                  # center outside (380, 1000) nm → 0.0

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_5nm),   1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_8nm),   0.5)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, bands_15nm),  0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, no_in_range), 0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, []),          0.0)

        # takes the best (min) resolution among filtered bands
        mixed = [(500, 100, 5), (700, 100, 12)]   # best resolution in range = 5nm → 1.0
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
        self.required_min_nm = 380.0   # nm — lower edge of required spectral window
        self.required_max_nm = 2500.0  # nm — upper edge of required spectral window
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
        # Returns min(1.0, overlap / FWHM) for the best-matching band.
        # overlap = intersection width of [center ± bw/2] with [required_min, required_max].

        # well inside required range: overlap (200nm) >> FWHM (10nm) → 1.0
        inside     = [(1000, 200, 10)]   # coverage=[900,1100]; overlap=200; 200/10=20 → 1.0

        # edge: overlap exactly == FWHM → 1.0
        edge_full  = [(370, 40, 10)]     # coverage=[350,390]; overlap=[380,390]=10; 10/10=1.0

        # partial: overlap (2nm) < FWHM (10nm) → partial credit
        partial    = [(376, 12, 10)]     # coverage=[370,382]; overlap=[380,382]=2; 2/10=0.2

        # entirely below required window → 0.0
        below      = [(200, 100, 5)]     # coverage=[150,250]; no overlap with [380,2500]

        # entirely above required window → 0.0
        above      = [(3000, 100, 5)]    # coverage=[2950,3050]; no overlap with [380,2500]

        # empty band list → 0.0
        empty      = []

        # best of multiple bands: first misses, second hits cleanly → 1.0
        multi      = [(200, 10, 5), (400, 20, 10)]  # second: coverage=[390,410]; overlap=20; 20/10=2→1.0

        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, inside),    1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, edge_full), 1.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, partial),   0.2)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, below),     0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, above),     0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, empty),     0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, multi),     1.0)

        # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, inside)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", inside)

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
        # Two-tier fixture: tier1 requires fine resolution AND full range (score=1.0);
        # tier2 requires only full range as a fallback (score=0.5).
        # Preference = tier_score × product(sub_req.prefs) for first tier where product > 0.
        self.res_req   = SpectralResolutionRequirement((380, 1000), [5], [1.0, 0.0])  # binary: ≤5nm → 1.0
        self.range_req = SpectralRangeRequirement(380.0, 2500.0)                       # graded: min(1, overlap/FWHM) for best band
        self.tiers = [
            {"score": 1.0, "requirements": [self.res_req, self.range_req]},  # fine res + full range
            {"score": 0.5, "requirements": [self.range_req]},                 # full range only (fallback)
        ]
        self.req = TieredSpectralRequirement(tiers=self.tiers)

    def test_constructor(self):
        self.assertIsInstance(self.req, TieredSpectralRequirement)
        self.assertEqual(self.req.req_type, RequirementTypes.SPECTRAL.value)
        self.assertEqual(self.req.attribute, ATTRIBUTE)
        self.assertEqual(self.req.strategy, SpectralPreferenceStrategies.TIERED.value)
        self.assertEqual(len(self.req.tiers), 2)

        # multiple requirements in a single tier is valid
        band_req = SpectralBandCountRequirement((380, 1000), [3], [0.0, 1.0])  # nm
        multi_req = TieredSpectralRequirement(tiers=[{
            "score": 1.0,
            "requirements": [self.res_req, band_req],
        }])
        self.assertEqual(len(multi_req.tiers[0]["requirements"]), 2)

        # invalid tiers: empty list
        self.assertRaises(AssertionError, TieredSpectralRequirement, tiers=[])

        # invalid tier: not a dict
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=["not_a_dict"])

        # invalid tier: missing "score"
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"requirements": [self.res_req]}])

        # invalid tier: missing "requirements"
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.0}])

        # invalid tier: score out of [0, 1]
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.5, "requirements": [self.res_req]}])

        # invalid tier: requirements not SpectralRequirement instances
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=[{"score": 1.0, "requirements": ["not_a_req"]}])

        # invalid tier: scores not in descending order
        self.assertRaises(AssertionError, TieredSpectralRequirement, tiers=[
            {"score": 0.5, "requirements": [self.range_req]},
            {"score": 1.0, "requirements": [self.res_req, self.range_req]},  # ascending → invalid
        ])

        # invalid id
        self.assertRaises(AssertionError, TieredSpectralRequirement,
                          tiers=self.tiers, id=123)
        self.assertRaises(ValueError, TieredSpectralRequirement,
                          tiers=self.tiers, id="123")

    def test_get_preference(self):
        # tier1 (score=1.0): res_req AND range_req  → value = 1.0 × res.pref × range.pref
        # tier2 (score=0.5): range_req only          → value = 0.5 × range.pref
        # returns value of first tier where value > 0

        # bands covering 380–2500 nm (lower edge=380, upper=2505) with 5nm resolution
        fine_in_range  = [(380, 10, 5), (2500, 10, 5)]   # res=5nm, covers 380–2505nm
        # bands covering 380–2500 nm but coarser resolution
        coarse_in_range = [(380, 10, 15), (2500, 10, 5)]  # res=15nm (>5nm threshold)
        # bands entirely outside the required range [380, 2500] nm
        fine_no_range  = [(200, 10, 5), (300, 10, 5)]

        # fine res + range ok: tier1 = 1.0 × 1.0 × 1.0 = 1.0 → return 1.0
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, fine_in_range),   1.0)
        # coarse res + range ok: tier1 = 1.0 × 0.0 × 1.0 = 0; tier2 = 0.5 × 1.0 = 0.5 → return 0.5
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, coarse_in_range), 0.5)
        # no range overlap: tier1 = 1.0 × 0.0 × 0.0 = 0; tier2 = 0.5 × 0.0 = 0 → return 0.0
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, fine_no_range),   0.0)
        self.assertAlmostEqual(self.req.calc_preference(ATTRIBUTE, []),              0.0)

        # sub-requirements with intermediate scores (non-binary): product scales the tier score
        inter_res = SpectralResolutionRequirement((380, 1000), [5, 10], [1.0, 0.5, 0.0])  # nm
        inter_req = TieredSpectralRequirement(tiers=[
            {"score": 1.0, "requirements": [inter_res, self.range_req]},
        ])
        # band (500, 240, *): coverage=[380,620]; overlap with [380,2500]=240nm >> FWHM → range_req=1.0
        fine_in_range2 = [(500, 240, 5), (2500, 10, 5)]  # VNIR res=5nm → inter_res 1.0; range 1.0
        coarse_in_range2 = [(500, 240, 8), (2500, 10, 5)]  # VNIR res=8nm → inter_res 0.5; range 1.0
        # fine_in_range2: tier1 = 1.0 × 1.0 × 1.0 = 1.0
        self.assertAlmostEqual(inter_req.calc_preference(ATTRIBUTE, fine_in_range2), 1.0)
        # coarse_in_range2: tier1 = 1.0 × 0.5 × 1.0 = 0.5 > 0 → return 0.5 (intermediate product)
        self.assertAlmostEqual(inter_req.calc_preference(ATTRIBUTE, coarse_in_range2), 0.5)

        # simultaneous sub-requirements: both must yield > 0 for the tier to pass
        tir_req  = SpectralBandCountRequirement((8000, 12000), [5], [0.0, 1.0])  # TIR range, nm
        mwir_req = SpectralBandCountRequirement((3000, 4500),  [1], [0.0, 1.0])  # MWIR range, nm
        multi_tier = TieredSpectralRequirement(tiers=[
            {"score": 1.0, "requirements": [tir_req, mwir_req]},
            {"score": 0.5, "requirements": [tir_req]},
        ])
        five_tir_mwir = [(8500,500,100),(9500,500,100),(10500,500,100),(11500,500,100),(12000,500,100),(3500,200,100)]
        five_tir_only = [(8500,500,100),(9500,500,100),(10500,500,100),(11500,500,100),(12000,500,100)]
        few_tir       = [(9000, 500, 100)]
        # tier1 requires BOTH TIR and MWIR; tier2 requires only TIR
        self.assertAlmostEqual(multi_tier.calc_preference(ATTRIBUTE, five_tir_mwir), 1.0)  # tier1 passes
        self.assertAlmostEqual(multi_tier.calc_preference(ATTRIBUTE, five_tir_only), 0.5)  # tier1 fails, tier2 passes
        self.assertAlmostEqual(multi_tier.calc_preference(ATTRIBUTE, few_tir),       0.0)  # both fail

        # wrong attribute
        self.assertRaises(AssertionError, self.req.calc_preference, 12345, fine_in_range)
        self.assertRaises(AssertionError, self.req.calc_preference, "wrong_attr", fine_in_range)

        # invalid band list
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, "not_a_list")
        self.assertRaises(AssertionError, self.req.calc_preference, ATTRIBUTE, [(500, 100)])

    def test_representation(self):
        expected = (
            "SpectralRequirement(strategy=TIERED, "
            "tiers=[(score=1.0, n_reqs=2), (score=0.5, n_reqs=1)])"
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
        self.assertEqual(len(d["tiers"][0]["requirements"]), 2)
        self.assertEqual(len(d["tiers"][1]["requirements"]), 1)
        self.assertEqual(d["tiers"][0]["requirements"][0]["strategy"],
                         SpectralPreferenceStrategies.RESOLUTION.value)
        self.assertEqual(d["tiers"][0]["requirements"][1]["strategy"],
                         SpectralPreferenceStrategies.RANGE.value)
        self.assertEqual(d["tiers"][1]["requirements"][0]["strategy"],
                         SpectralPreferenceStrategies.RANGE.value)

    def test_from_dict(self):
        req_dict = {
            "req_type": RequirementTypes.SPECTRAL.value,
            "attribute": ATTRIBUTE,
            "strategy": SpectralPreferenceStrategies.TIERED.value,
            "tiers": [
                {
                    "score": 1.0,
                    "requirements": [
                        {
                            "req_type": RequirementTypes.SPECTRAL.value,
                            "attribute": ATTRIBUTE,
                            "strategy": SpectralPreferenceStrategies.RESOLUTION.value,
                            "wavelength_range": [380, 1000],
                            "thresholds": [5],
                            "scores": [1.0, 0.0],
                        },
                        {
                            "req_type": RequirementTypes.SPECTRAL.value,
                            "attribute": ATTRIBUTE,
                            "strategy": SpectralPreferenceStrategies.RANGE.value,
                            "required_min_nm": 380.0,
                            "required_max_nm": 2500.0,
                        },
                    ],
                },
                {
                    "score": 0.5,
                    "requirements": [
                        {
                            "req_type": RequirementTypes.SPECTRAL.value,
                            "attribute": ATTRIBUTE,
                            "strategy": SpectralPreferenceStrategies.RANGE.value,
                            "required_min_nm": 380.0,
                            "required_max_nm": 2500.0,
                        },
                    ],
                },
            ],
        }
        # class method
        r = TieredSpectralRequirement.from_dict(req_dict)
        self.assertIsInstance(r, TieredSpectralRequirement)
        self.assertEqual(len(r.tiers), 2)
        self.assertAlmostEqual(r.tiers[0]["score"], 1.0)
        self.assertAlmostEqual(r.tiers[1]["score"], 0.5)
        self.assertIsInstance(r.tiers[0]["requirements"][0], SpectralResolutionRequirement)
        self.assertIsInstance(r.tiers[0]["requirements"][1], SpectralRangeRequirement)
        self.assertIsInstance(r.tiers[1]["requirements"][0], SpectralRangeRequirement)
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
