def dump_free
    print $arg0->path
    set $pgt = $arg0->pg_table
    set $next = $pgt->pg_free
    while $next != 0
        print $next
        set $pg = &$pgt->pg_pages[$next]
        print/x *$pg
        set $next = $pg->pg_next
    end
end


def dump_all
    print $arg0->path
    set $pgt = $arg0->pg_table
    set $next = 1
    while $next < $pgt->pg_count
        print $next
        set $pg = &$pgt->pg_pages[$next]
	print $pg
        print/x *$pg
        set $next = $next + $pg->pg_count
    end
end

def pnode
    if $argc > 1
        print/x ((bxt_node_t)((ods_obj_t)$arg0)->as.ptr)->$arg1
    else
        print/x *((bxt_node_t)((ods_obj_t)$arg0)->as.ptr)


