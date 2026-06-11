"""
cache/clock.py
--------------
Clock (Second Chance) cache replacement policy.

The Clock algorithm approximates LRU without the overhead of reordering
on every access. Pages sit in a circular buffer. Each page has a reference
bit. When eviction is needed, a clock hand sweeps the circle:
    - bit = 1 → give a second chance: clear bit, advance hand
    - bit = 0 → evict this page, insert new page here, advance hand

This is cheaper than LRU (no move_to_end on every hit) while still
outperforming FIFO by sparing recently used pages one extra cycle.
"""

from .base import CacheBase, CacheStats


class Clock(CacheBase):
    """
    Clock (Second Chance) cache replacement policy.

    Internal data structures:
        _frames  (list): Circular buffer of fixed size = capacity.
                         Each slot holds a page number or None.
        _ref_bits (list): Reference bit per slot. 1 = recently used,
                          0 = not recently used (candidate for eviction).
        _hand    (int):  Current position of the clock hand in _frames.
                         Always in range [0, capacity - 1].

    Clock hand movement:
        The hand only advances during eviction. It never resets to 0
        between accesses — it resumes from wherever it last stopped,
        giving the algorithm its circular (clock-like) behaviour.

    Args:
        capacity (int): Maximum number of pages the cache can hold.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._frames: list = [None] * capacity    # circular buffer
        self._ref_bits: list = [0] * capacity     # reference bits
        self._hand: int = 0                        # clock hand position
        self._page_to_slot: dict = {}              # page → slot index for O(1) lookup

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        HIT  — page is in cache. Set its reference bit to 1.
        MISS — page is not in cache. Find a victim using the clock
               algorithm, evict it, and insert the new page.

        Clock eviction algorithm:
            1. Look at the frame at the hand position.
            2. If reference bit = 1: clear it to 0, advance hand, repeat.
            3. If reference bit = 0: evict this page, place new page here,
               set reference bit = 1, advance hand.

        Args:
            page (int): The page number being requested.

        Returns:
            bool: True if HIT, False if MISS.
        """
        if page in self._page_to_slot:
            # ── HIT ──
            self.stats.hits += 1
            slot = self._page_to_slot[page]
            self._ref_bits[slot] = 1
            return True

        # ── MISS ──
        self.stats.misses += 1

        if len(self._page_to_slot) < self.capacity:
            # Cache not yet full — find the next empty slot
            slot = self._find_empty_slot()
        else:
            # Cache full — run clock hand to find a victim
            slot = self._find_victim()
            evicted_page = self._frames[slot]
            del self._page_to_slot[evicted_page]
            self.stats.evictions += 1

        # Insert new page into the chosen slot
        self._frames[slot] = page
        self._ref_bits[slot] = 1
        self._page_to_slot[page] = slot
        self._hand = (slot + 1) % self.capacity

        return False

    def _find_empty_slot(self) -> int:
        """
        Find the next empty (None) slot in the circular buffer.

        Used only during the initial warmup phase before the cache
        reaches full capacity. Scans from the current hand position.

        Returns:
            int: Index of the first empty slot found.
        """
        for i in range(self.capacity):
            slot = (self._hand + i) % self.capacity
            if self._frames[slot] is None:
                return slot
        return self._hand  # fallback, should not reach here

    def _find_victim(self) -> int:
        """
        Run the clock hand to find a victim slot for eviction.

        Sweeps the circular buffer starting at the current hand position:
            - Slot with reference bit = 1: clear bit to 0, advance hand.
            - Slot with reference bit = 0: this is the victim, return slot.

        In the worst case (all bits = 1), the hand makes one full
        revolution clearing bits, then evicts on the second pass.

        Returns:
            int: Index of the slot chosen for eviction.
        """
        while True:
            if self._ref_bits[self._hand] == 0:
                # Found victim
                victim_slot = self._hand
                return victim_slot
            else:
                # Second chance — clear bit and move on
                self._ref_bits[self._hand] = 0
                self._hand = (self._hand + 1) % self.capacity

    def reset(self):
        """
        Clear all pages, reference bits, and reset stats to zero.

        Resets the clock hand to position 0 and clears all internal
        state, ready for a fresh simulation run.
        """
        self._frames = [None] * self.capacity
        self._ref_bits = [0] * self.capacity
        self._hand = 0
        self._page_to_slot.clear()
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """
        Return current performance statistics.

        Returns:
            CacheStats: Dataclass with hits, misses, and evictions.
        """
        return self.stats