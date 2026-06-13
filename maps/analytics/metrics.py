def congestion_score(waiting_time, throughput):
    return (
        0.7 * waiting_time
        - 0.3 * throughput
    )


def traffic_efficiency(waiting_time, throughput):

    if waiting_time <= 0:
        return 100

    return (throughput / waiting_time) * 100


def emergency_priority_score(priority_counts):
    """
    priority_counts:
    {
        "ambulance": x,
        "firetruck": y,
        "police": z
    }
    """

    return (
        priority_counts.get("ambulance", 0) * 3 +
        priority_counts.get("firetruck", 0) * 2 +
        priority_counts.get("police", 0) * 1
    )