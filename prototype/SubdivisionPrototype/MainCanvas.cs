using System;
using Gtk;
using Cairo;

namespace SubdivisionPrototype {

[System.ComponentModel.ToolboxItem(true)]
public class MainCanvas : Gtk.DrawingArea {

    const int CURSOR_SIZE = 10;
    const int PADDING = 50;

    private Quad root;

    private double cursor_x = PADDING / 2;
    private double cursor_y = PADDING / 2;

    private double upper_left_x;
    private double upper_left_y;
    private double lower_right_x;
    private double lower_right_y;

    public MainCanvas() {
        this.AddEvents((int)(Gdk.EventMask.PointerMotionMask | Gdk.EventMask.ButtonPressMask));
        this.MotionNotifyEvent += on_motion_notify_event;
    }

    private void on_motion_notify_event(object o, MotionNotifyEventArgs args) {
        if ((args.Event.State & Gdk.ModifierType.Button1Mask) == 0) {
            return;
        }

        this.cursor_x = args.Event.X;
        this.cursor_y = args.Event.Y;

        this.QueueDraw();
    }

    private double trans_x(double x) {
        return upper_left_x + x * (lower_right_x - upper_left_x);
    }

    private double trans_y(double y) {
        return upper_left_y + y * (lower_right_y - upper_left_y);
    }

    private double inv_trans_x(double x) {
        double ret = (x - upper_left_x) / (lower_right_x - upper_left_x);
        if (ret < 0) {
            return 0;
        } else if (ret > 1) {
            return 1;
        } else {
            return ret;
        }
    }

    private double inv_trans_y(double y) {
        double ret = (y - upper_left_y) / (lower_right_y - upper_left_y);
        if (ret < 0) {
            return 0;
        } else if (ret > 1) {
            return 1;
        } else {
            return ret;
        }
    }

    private void draw_quad_bg(Context ctx, Quad quad) {
        int base_coord_x = quad.get_base_coord_x();
        int base_coord_y = quad.get_base_coord_y();
        int quad_length = Quad.quad_length(quad.get_level());
        int max_coord = Quad.max_coord();

        double p1x = trans_x((double)base_coord_x / (double)max_coord);
        double p1y = trans_y((double)base_coord_y / (double)max_coord);
        double p2x = trans_x((double)(base_coord_x + quad_length) / (double)max_coord);
        double p2y = trans_y((double)(base_coord_y + quad_length) / (double)max_coord);

        if (!quad.is_subdivided() && quad.in_subdivision_range() && !quad.can_subdivide()) {
            // This is a quad in range of subdivision, but cannot subdivide because its neighbours aren't subdivided 
            // enough
            double r = 0.2 + ((double)quad.get_level() / (double)Quad.max_subdivision) * 0.4;
            double g = 0.2 + ((double)quad.get_level() / (double)Quad.max_subdivision) * 0.4;
            double b = 0.4 + ((double)quad.get_level() / (double)Quad.max_subdivision) * 0.6;
            ctx.Rectangle(new Rectangle(p1x, p1y, p2x - p1x, p2y - p1y));
            ctx.SetSourceColor(new Color(r, g, b));
            ctx.Fill();
        } else {
            ctx.Rectangle(new Rectangle(p1x, p1y, p2x - p1x, p2y - p1y));
            ctx.SetSourceColor(new Color(0.9, 0.8, 0.8));
            ctx.Fill();
        }
    }

    private void draw_quad_lines(Context ctx, Quad quad) {
        int base_coord_x = quad.get_base_coord_x();
        int base_coord_y = quad.get_base_coord_y();
        int quad_length = Quad.quad_length(quad.get_level());
        int half_quad_length = quad_length / 2;
        int max_coord = Quad.max_coord();

        double p1x = trans_x((double)base_coord_x / (double)max_coord);
        double p1y = trans_y((double)base_coord_y / (double)max_coord);
        double p2x = trans_x((double)(base_coord_x + quad_length) / (double)max_coord);
        double p2y = trans_y((double)(base_coord_y + quad_length) / (double)max_coord);

        if (quad == this.root) {
            // Draw a rectangle around the root
            ctx.Rectangle(new Rectangle(p1x, p1y, p2x - p1x, p2y - p1y));
            ctx.SetSourceColor(new Color(0.0, 0.0, 0.0));
            ctx.Stroke();
        }

        if (quad.is_subdivided()) {
            p1x = trans_x((double)(base_coord_x + half_quad_length) / (double)max_coord);
            p1y = trans_y((double)base_coord_y / (double)max_coord);
            p2x = trans_x((double)(base_coord_x + half_quad_length) / (double)max_coord);
            p2y = trans_y((double)(base_coord_y + quad_length) / (double)max_coord);
            ctx.MoveTo(p1x, p1y);
            ctx.LineTo(p2x, p2y);
            ctx.Stroke();

            p1x = trans_x((double)base_coord_x / (double)max_coord);
            p1y = trans_y((double)(base_coord_y + half_quad_length) / (double)max_coord);
            p2x = trans_x((double)(base_coord_x + quad_length) / (double)max_coord);
            p2y = trans_y((double)(base_coord_y + half_quad_length) / (double)max_coord);
            ctx.MoveTo(p1x, p1y);
            ctx.LineTo(p2x, p2y);
            ctx.Stroke();
        }
    }

    private void draw_quad_bg_rec(Context ctx, Quad quad) {
        draw_quad_bg(ctx, quad);
        Quad[] children = quad.get_children();
        if (children != null) {
            foreach (Quad q in children) {
                this.draw_quad_bg_rec(ctx, q);   
            }
        }
    }

    private void draw_quad_lines_rec(Context ctx, Quad quad) {
        draw_quad_lines(ctx, quad);
        Quad[] children = quad.get_children();
        if (children != null) {
            foreach (Quad q in children) {
                this.draw_quad_lines_rec(ctx, q);   
            }
        }
    }

    private void draw_cursor(Context ctx) {
        double p1x = this.cursor_x - CURSOR_SIZE / 2;
        double p1y = this.cursor_y;
        double p2x = this.cursor_x + CURSOR_SIZE / 2;
        double p2y = this.cursor_y;
        ctx.MoveTo(p1x, p1y);
        ctx.LineTo(p2x, p2y);
        ctx.Stroke();

        p1x = this.cursor_x;
        p1y = this.cursor_y - CURSOR_SIZE / 2;
        p2x = this.cursor_x;
        p2y = this.cursor_y + CURSOR_SIZE / 2;
        ctx.MoveTo(p1x, p1y);
        ctx.LineTo(p2x, p2y);
        ctx.Stroke();
    }

    private void draw(Context ctx) {
        this.draw_quad_bg_rec(ctx, root);
        this.draw_quad_lines_rec(ctx, root);
        this.draw_cursor(ctx);
    }

    protected override bool OnExposeEvent(Gdk.EventExpose args)
    {
        int width = this.Allocation.Width;
        int height = this.Allocation.Height;

        int size = Math.Min(width, height) - 2 * PADDING;
        if (size <= 0) {
            return true;
        } else {
            this.upper_left_x = PADDING;
            this.upper_left_y = PADDING;
            this.lower_right_x = PADDING + size;
            this.lower_right_y = PADDING + size;
        }
        if (root == null) {
            root = new Quad();
        } else {
            Quad.centre_pos = new Point2(
                (float)inv_trans_x(this.cursor_x),
                (float)inv_trans_y(this.cursor_y));
            root.check_subdivision();
        }
        using (Context ctx = Gdk.CairoHelper.Create (args.Window)) {
            draw(ctx);
        }

        return true;
    }
}

}

