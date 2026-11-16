"""conductor-providers: conductor in the frameworks a host already uses.

``conductor_providers.fastapi.run_router`` serves runs over HTTP: start one,
follow its events, answer it, resume it and cancel it, on any number of
workers sharing one store.
"""
