# Phase 5 Manual Browser Test Plan

Manual browser tests have not yet been run. Do not treat this file as evidence of successful browser behavior until each result is recorded.

For each test, record:

- page URL or local fixture
- API mode: local model or mock mode
- browser version
- zoom level
- result: pass/fail
- notes and screenshots if needed

## Required Tests

1. Static page with one image.
2. Static page with several images.
3. Long page with many images.
4. Lazy-loaded images.
5. Infinite-scroll page.
6. Same image displayed multiple times.
7. Responsive image using `srcset`.
8. `object-fit: contain`.
9. `object-fit: cover`.
10. Page zoom at 80%, 100%, and 125%.
11. Window resize.
12. Page scroll after overlays appear.
13. Multiple faces.
14. No-face image.
15. API offline.
16. API restart.
17. Stop scan while requests are queued.
18. Disable current site.
19. Clear cache.
20. Extension reload.
21. Service-worker suspension and recovery.

## Expected Behavior

- Manual right-click analysis still works.
- Page scanning begins only after a user action or after the user has enabled the current site.
- Disabled sites do not page-scan.
- Global emergency stop prevents page scanning.
- Image errors never become `Likely Real`.
- No-face results show no face verdict.
- Multiple faces render independently.
- Overlays stay aligned while scrolling, resizing, and using `object-fit: contain` or `cover`.
- Mock mode is visibly marked.
- No raw image bytes or face crops are stored.
