"""Unit filter class."""


class UnitFilter:
    """Callable for filtering units based on quality and brain area."""

    _BRAIN_AREA_TO_PROBE_NAMES = {
        "DMFC": ("s0",),
        "FEF": ("vprobe0", "vprobe1"),
    }

    def __init__(
        self,
        brain_areas: tuple = ("DMFC", "FEF"),
        qualities: tuple = ("good", "mua"),
    ):
        """Constructor.

        Args:
            brain_areas: Tuple of brain areas to filter units by.
            qualities: Tuple of qualities to filter units by.
        """
        self._brain_areas = brain_areas
        self._qualities = qualities

        # Create probe names
        self._probe_names = []
        for brain_area in brain_areas:
            self._probe_names.extend(
                self._BRAIN_AREA_TO_PROBE_NAMES[brain_area]
            )

    def __call__(self, df_units):
        """Filter units.

        Args:
            df_units: DataFrame containing unit data.

        Returns:
            Filtered DataFrame with units that match the specified brain areas
            and qualities, and have enough trials if specified.
        """
        # Filter by quality
        df_units = df_units[df_units.quality.isin(self._qualities)]
        # Filter by brain area
        df_units = df_units[df_units.probe.isin(self._probe_names)]

        # Filter significant
        df_units = df_units[df_units.significant]

        return df_units
