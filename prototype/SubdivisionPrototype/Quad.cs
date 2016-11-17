using System;

namespace SubdivisionPrototype {

public class Quad {
    private Quad north;
    private Quad south;
    private Quad east;
    private Quad west;

    private Quad parent;

    private Quad[] children;

    private int base_coord_x;
    private int base_coord_y;

    private int level;
    private QuadPos pos;

    public const int QUAD_MESH_SIZE = 14;
    public static int max_subdivision = 5;

    public Quad() {
    }

    public static Point2 centre_pos;

    public static int quad_length(int level) {
        return (1 << (max_subdivision - level)) * QUAD_MESH_SIZE;
    }

    public static int max_coord() {
        return (1 << max_subdivision) * QUAD_MESH_SIZE;
    }

    public int get_base_coord_x() {
        return this.base_coord_x;
    }

    public int get_base_coord_y() {
        return this.base_coord_y;
    }

    public int get_level() {
        return this.level;
    }

    public Quad[] get_children() {
        return this.children;
    }

    private Point2 mid_coord_pos() {
        int max_coord = Quad.max_coord();
        int quad_length = Quad.quad_length(this.level);
        int half_quad_length = quad_length / 2;
        int mid_coord_x = base_coord_x + half_quad_length;
        int mid_coord_y = base_coord_y + half_quad_length;

        float x = (float)mid_coord_x / (float)max_coord;
        float y = (float)mid_coord_y / (float)max_coord;

        Point2 ret = new Point2();
        ret.x = x;
        ret.y = y;
        return ret;
    }

    public bool in_subdivision_range() {
        if (this.level == max_subdivision) {
            return false;
        }

        float extra = (float)Math.Pow(1.05, (double)(Quad.max_subdivision - this.level));
        Point2 mid_coord_pos = this.mid_coord_pos();
        float dst = centre_pos.distance(mid_coord_pos);
        float range = extra * 1.5f * ((float)Quad.quad_length(this.level) / (float)Quad.max_coord());
        return dst <= range;
    }

    private bool in_collapse_range() {
        float extra = (float)Math.Pow(1.05, (double)(Quad.max_subdivision - this.level));
        Point2 mid_coord_pos = this.mid_coord_pos();
        float dst = centre_pos.distance(mid_coord_pos);
        float range = extra * 2 * 1.5f * ((float)Quad.quad_length(this.level) / (float)Quad.max_coord());
        return dst >= range;
    }

    public bool can_subdivide() {
        Quad indirect1 = null, indirect2 = null;
        switch (this.pos) {
            case QuadPos.UPPER_LEFT:
                indirect1 = north;
                indirect2 = west;
                break;
            case QuadPos.UPPER_RIGHT:
                indirect1 = north;
                indirect2 = east;
                break;
            case QuadPos.LOWER_LEFT:
                indirect1 = south;
                indirect2 = west;
                break;
            case QuadPos.LOWER_RIGHT:
                indirect1 = south;
                indirect2 = east;
                break;
            default:
                return true;
        }
        // On an actual quadsphere, the null checks wouldn't be needed
        if (indirect1 != null && !indirect1.is_subdivided()) {
            return false;
        } else if (indirect2 != null && !indirect2.is_subdivided()) {
            return false;
        } else {
            return true;
        }
    }

    public bool can_collapse() {
        Quad direct_north = this.direct_north();
        Quad direct_south = this.direct_south();
        Quad direct_east = this.direct_east();
        Quad direct_west = this.direct_west();

        if (direct_north != null && direct_north.is_subdivided()) {
            Quad q1 = direct_north.get_child(QuadPos.LOWER_LEFT);
            Quad q2 = direct_north.get_child(QuadPos.LOWER_RIGHT);
            if (q1.is_subdivided() || q2.is_subdivided()) {
                return false;
            }
        }

        if (direct_south != null && direct_south.is_subdivided()) {
            Quad q1 = direct_south.get_child(QuadPos.UPPER_LEFT);
            Quad q2 = direct_south.get_child(QuadPos.UPPER_RIGHT);
            if (q1.is_subdivided() || q2.is_subdivided()) {
                return false;
            }
        }

        if (direct_east != null && direct_east.is_subdivided()) {
            Quad q1 = direct_east.get_child(QuadPos.UPPER_LEFT);
            Quad q2 = direct_east.get_child(QuadPos.LOWER_LEFT);
            if (q1.is_subdivided() || q2.is_subdivided()) {
                return false;
            }
        }

        if (direct_west != null && direct_west.is_subdivided()) {
            Quad q1 = direct_west.get_child(QuadPos.UPPER_RIGHT);
            Quad q2 = direct_west.get_child(QuadPos.LOWER_RIGHT);
            if (q1.is_subdivided() || q2.is_subdivided()) {
                return false;
            }
        }

        return true;
    }

    public bool is_subdivided() {
        return this.children != null;
    }

    public Quad get_child(QuadPos pos) {
        return (this.children != null) ? this.children[pos.to_idx()] : null;
    }

    private Quad direct_north() {
        if (this.pos == QuadPos.LOWER_LEFT || this.pos == QuadPos.LOWER_RIGHT) {
            return this.north;
        } else if (this.pos == QuadPos.UPPER_LEFT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.north != null) return this.north.get_child(QuadPos.LOWER_LEFT);
        } else if (this.pos == QuadPos.UPPER_RIGHT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.north != null) return this.north.get_child(QuadPos.LOWER_RIGHT);
        }
        return null;
    }

    private Quad direct_south() {
        if (this.pos == QuadPos.UPPER_LEFT || this.pos == QuadPos.UPPER_RIGHT) {
            return this.south;
        } else if (this.pos == QuadPos.LOWER_LEFT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.south != null) return this.south.get_child(QuadPos.UPPER_LEFT);
        } else if (this.pos == QuadPos.LOWER_RIGHT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.south != null) return this.south.get_child(QuadPos.UPPER_RIGHT);
        }
        return null;
    }

    private Quad direct_east() {
        if (this.pos == QuadPos.UPPER_LEFT || this.pos == QuadPos.LOWER_LEFT) {
            return this.east;
        } else if (this.pos == QuadPos.UPPER_RIGHT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.east != null) return this.east.get_child(QuadPos.UPPER_LEFT);
        } else if (this.pos == QuadPos.LOWER_RIGHT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.east != null) return this.east.get_child(QuadPos.LOWER_LEFT);
        }
        return null;
    }

    private Quad direct_west() {
        if (this.pos == QuadPos.UPPER_RIGHT || this.pos == QuadPos.LOWER_RIGHT) {
            return this.west;
        } else if (this.pos == QuadPos.UPPER_LEFT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.west != null) return this.west.get_child(QuadPos.UPPER_RIGHT);
        } else if (this.pos == QuadPos.LOWER_LEFT) {
            // On an actual quadsphere, the null check wouldn't be needed
            if (this.west != null) return this.west.get_child(QuadPos.LOWER_RIGHT);
        }
        return null;
    }

    private void subdivide() {
        int quad_length = Quad.quad_length(this.level);
        int half_quad_length = quad_length / 2;

        Quad upper_left = new Quad();
        Quad upper_right = new Quad();
        Quad lower_left = new Quad();
        Quad lower_right = new Quad();

        Quad direct_north = this.direct_north();
        Quad direct_south = this.direct_south();
        Quad direct_east = this.direct_east();
        Quad direct_west = this.direct_west();

        upper_left.pos = QuadPos.UPPER_LEFT;
        upper_left.north = direct_north;
        upper_left.east = upper_right;
        upper_left.south = lower_left;
        upper_left.west = direct_west;
        upper_left.parent = this;
        upper_left.level = this.level + 1;
        upper_left.base_coord_x = this.base_coord_x;
        upper_left.base_coord_y = this.base_coord_y;

        upper_right.pos = QuadPos.UPPER_RIGHT;
        upper_right.north = direct_north;
        upper_right.east = direct_east;
        upper_right.south = lower_right;
        upper_right.west = upper_left;
        upper_right.parent = this;
        upper_right.level = this.level + 1;
        upper_right.base_coord_x = this.base_coord_x + half_quad_length;
        upper_right.base_coord_y = this.base_coord_y;

        lower_left.pos = QuadPos.LOWER_LEFT;
        lower_left.north = upper_left;
        lower_left.east = lower_right;
        lower_left.south = direct_south;
        lower_left.west = direct_west;
        lower_left.parent = this;
        lower_left.level = this.level + 1;
        lower_left.base_coord_x = this.base_coord_x;
        lower_left.base_coord_y = this.base_coord_y + half_quad_length;

        lower_right.pos = QuadPos.LOWER_RIGHT;
        lower_right.north = upper_right;
        lower_right.east = direct_east;
        lower_right.south = direct_south;
        lower_right.west = lower_left;
        lower_right.parent = this;
        lower_right.level = this.level + 1;
        lower_right.base_coord_x = this.base_coord_x + half_quad_length;
        lower_right.base_coord_y = this.base_coord_y + half_quad_length;

        this.children = new Quad[4];
        this.children[QuadPos.UPPER_LEFT.to_idx()] = upper_left;
        this.children[QuadPos.UPPER_RIGHT.to_idx()] = upper_right;
        this.children[QuadPos.LOWER_LEFT.to_idx()] = lower_left;
        this.children[QuadPos.LOWER_RIGHT.to_idx()] = lower_right;
    }

    private void collapse() {
        this.children = null;
    }

    public void check_subdivision() {
        if (!this.is_subdivided() && this.level <= Quad.max_subdivision && this.in_subdivision_range() && this.can_subdivide()) {
            this.subdivide();
        } else if (this.is_subdivided() && this.in_collapse_range() && this.can_collapse()) {
            this.collapse();
        } else if (this.is_subdivided()) {
            foreach (Quad q in this.children) {
                q.check_subdivision();
            }
        }
    }
}

}

