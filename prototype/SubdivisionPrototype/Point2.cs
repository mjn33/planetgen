using System;

namespace SubdivisionPrototype {

public struct Point2 {
    public float x;
    public float y;

    public Point2(float x, float y) {
        this.x = x;
        this.y = y;
    }

    public float distance(Point2 other) {
        float dx = this.x - other.x;
        float dy = this.y - other.y;
        return (float)Math.Sqrt((double)(dx * dx + dy  * dy));
    }
}

}

