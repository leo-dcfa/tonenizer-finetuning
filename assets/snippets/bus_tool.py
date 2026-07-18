class BusTimes(BaseModel):
    """Next buses from a Noosa stop."""

    stop: str = Field(description="stop name")
    route: str | None = None


@tool(BusTimes)
def bus_times(args):
    return translink.departures(args.stop)
