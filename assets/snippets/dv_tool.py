class SupportSearch(BaseModel):
    """DFV support services near a suburb."""

    suburb: str
    urgent: bool = Field(description="danger")


@tool(SupportSearch)
def dv_support(args):
    if args.urgent:
        return "Call 000 — 1800RESPECT"
    return services.search("dfv", args.suburb)
