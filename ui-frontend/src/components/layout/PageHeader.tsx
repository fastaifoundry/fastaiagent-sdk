interface PageHeaderProps {
  title: string;
  description?: string;
  children?: React.ReactNode;
}

/**
 * The one <h1> on a screen. Uses the named type roles from index.css rather
 * than raw sizes, so every page's title matches and the classic skin can
 * restore its larger proportions with a single override.
 */
export function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div>
        <h1 className="fa-page-title">{title}</h1>
        {description && <p className="fa-page-subtitle">{description}</p>}
      </div>
      {children}
    </div>
  );
}
