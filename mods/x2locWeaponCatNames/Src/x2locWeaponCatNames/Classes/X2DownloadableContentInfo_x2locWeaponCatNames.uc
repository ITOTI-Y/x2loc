//---------------------------------------------------------------------------------------
//  Names the weapon categories that mods add, in the "valid weapons" line which
//  Musashi's Mods Fixes appends to ability descriptions.
//
//  That mod names a category through X2WeaponTemplate.GetLocalizedCategory, whose
//  listeners only register once a campaign loads, while it builds the line when
//  templates are created. A mod-added category therefore shows as its internal
//  name, lowercased with underscores turned into spaces. This runs after it and
//  replaces each such name with what the category's own weapons call themselves:
//  their AbilityDescName, or failing that their FriendlyName, both read from the
//  weapon mods' localization. Nothing is hard-coded, so new mods are covered as
//  soon as their localization is.
//---------------------------------------------------------------------------------------
class X2DownloadableContentInfo_x2locWeaponCatNames extends X2DownloadableContentInfo;

// One way a category's weapons name themselves, and how many do.
struct CategoryName
{
	var string Shown;			// as Musashi's Mods Fixes shows the unnamed category
	var string Name;
	var bool bDescName;			// from AbilityDescName, the generic weapon name
	var int Count;
};

static event OnPostTemplatesCreated()
{
	local X2AbilityTemplateManager Manager;
	local X2DataTemplate DataTemplate;
	local X2AbilityTemplate AbilityTemplate;
	local array<CategoryName> Names;

	Names = CollectNames();
	Manager = class'X2AbilityTemplateManager'.static.GetAbilityTemplateManager();
	foreach Manager.IterateTemplates(DataTemplate)
	{
		AbilityTemplate = X2AbilityTemplate(DataTemplate);
		if (AbilityTemplate != none)
		{
			AbilityTemplate.LocLongDescription = NameCategories(AbilityTemplate.LocLongDescription, Names);
		}
	}
}

static function array<CategoryName> CollectNames()
{
	local X2ItemTemplateManager Manager;
	local X2DataTemplate DataTemplate;
	local X2WeaponTemplate WeaponTemplate;
	local array<CategoryName> Names;
	local CategoryName Entry;
	local int i;

	Manager = class'X2ItemTemplateManager'.static.GetItemTemplateManager();
	foreach Manager.IterateTemplates(DataTemplate)
	{
		WeaponTemplate = X2WeaponTemplate(DataTemplate);
		if (WeaponTemplate == none || WeaponTemplate.WeaponCat == '')
		{
			continue;
		}
		Entry.Shown = Locs(Repl(string(WeaponTemplate.WeaponCat), "_", " "));
		// The getters stand in "Error! ..." for a missing name.
		Entry.Name = WeaponTemplate.GetItemAbilityDescName();
		Entry.bDescName = Left(Entry.Name, 7) != "Error! ";
		if (!Entry.bDescName)
		{
			Entry.Name = WeaponTemplate.GetItemFriendlyNameNoStats();
		}
		if (Left(Entry.Name, 7) == "Error! ")
		{
			continue;
		}
		for (i = 0; i < Names.Length; i++)
		{
			if (Names[i].Shown == Entry.Shown && Names[i].Name == Entry.Name && Names[i].bDescName == Entry.bDescName)
			{
				Names[i].Count++;
				break;
			}
		}
		if (i == Names.Length)
		{
			Entry.Count = 1;
			Names.AddItem(Entry);
		}
	}
	return Names;
}

// The line is "\n" $ GetColoredText(Names, eUIState_Warning), appended last.
static function string NameCategories(string Description, const out array<CategoryName> Names)
{
	local string Prefix, Suffix, Line, Named;
	local array<string> Shown;
	local int Start, i;

	Prefix = "\n<font color='#" $ class'UIUtilities_Colors'.const.WARNING_HTML_COLOR $ "'>";
	Suffix = "</font>";
	Start = InStr(Description, Prefix, true);
	if (Start == INDEX_NONE || Right(Description, Len(Suffix)) != Suffix)
	{
		return Description;
	}

	Line = Mid(Description, Start + Len(Prefix), Len(Description) - Start - Len(Prefix) - Len(Suffix));
	ParseStringIntoArray(Line, Shown, ", ", false);
	for (i = 0; i < Shown.Length; i++)
	{
		if (i > 0)
		{
			Named $= ", ";
		}
		Named $= BestName(Shown[i], Names);
	}
	return Left(Description, Start + Len(Prefix)) $ Named $ Suffix;
}

// A generic AbilityDescName beats a FriendlyName; then the most common wins.
// A name that matches no category, such as one already named, stays as it is.
static function string BestName(string Shown, const out array<CategoryName> Names)
{
	local int i, Best;

	Best = INDEX_NONE;
	for (i = 0; i < Names.Length; i++)
	{
		if (Names[i].Shown != Shown)
		{
			continue;
		}
		if (Best == INDEX_NONE
			|| (Names[i].bDescName && !Names[Best].bDescName)
			|| (Names[i].bDescName == Names[Best].bDescName && Names[i].Count > Names[Best].Count))
		{
			Best = i;
		}
	}
	return Best == INDEX_NONE ? Shown : Names[Best].Name;
}
